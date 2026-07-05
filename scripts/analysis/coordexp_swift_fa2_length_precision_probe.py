#!/usr/bin/env python3
"""Probe FA2 packed-row length numerical invariance for CoordExp-swift."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.config.loader import load_train_config
from src.data.jsonl import load_raw_examples
from src.packing.planner import PackedSequence, plan_packed_sequences
from src.qwen.encoding import encode_rendered_example
from src.qwen.forward import build_qwen_forward_inputs, run_qwen_forward
from src.qwen.loading import load_qwen_components
from src.qwen.positions import build_qwen_position_inputs
from src.templates import render_example


DEFAULT_CONFIG = (
    "configs/coordexp_swift/smoke/length_isolation/"
    "physical_length_12k_ebs64_1step.yaml"
)


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    resolved = load_train_config(args.config)
    output_dir = args.output_dir / datetime.now(timezone.utc).strftime(
        "fa2_length_precision_%Y%m%dT%H%M%SZ"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    components = load_qwen_components(resolved.config, load_model=True)
    model = components.model
    if model is None:
        raise RuntimeError("load_qwen_components(load_model=True) returned no model")
    model.to(device)
    model.eval()

    encoded = _load_encoded_examples(
        resolved.config,
        components=components,
        example_count=args.example_count,
    )
    target = encoded[args.target_index]
    variants = _build_sweep_variants(
        encoded,
        target_index=args.target_index,
        target_lengths=tuple(args.sweep_lengths),
    )
    logits_by_variant: dict[str, torch.Tensor] = {}
    receipts: dict[str, Any] = {}
    selected_offsets: tuple[int, ...] | None = None
    with torch.inference_mode():
        for name, variant in variants.items():
            target_segment = _target_segment(variant.pack, variant.target_example_id)
            offsets = _select_offsets(
                target_segment.length,
                positions_per_target=args.positions_per_target,
            )
            if selected_offsets is None:
                selected_offsets = offsets
            elif selected_offsets != offsets:
                raise RuntimeError("target offset selection diverged across variants")
            positions = tuple(target_segment.start + offset for offset in offsets)
            forward_inputs = build_qwen_forward_inputs(
                variant.pack,
                variant.examples,
                build_qwen_position_inputs(variant.pack, variant.examples),
                logits_to_keep_positions=positions,
                device=device,
                fa2_branch_proof_policy="every_forward",
            )
            result = run_qwen_forward(
                model,
                forward_inputs,
                expected_vocab_size=components.model_identity.text_vocab_size,
                capture_fa2_branch=True,
                require_fa2_branch_proof=True,
            )
            logits = _logits_2d(result.logits).detach().cpu()
            logits_by_variant[name] = logits
            receipts[name] = {
                "variant": variant.to_artifact_dict(),
                "selected_offsets": list(offsets),
                "selected_absolute_positions": list(positions),
                "receipt": result.receipt.to_artifact_dict(),
            }
            del result, forward_inputs, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    comparisons = _build_comparisons(logits_by_variant)
    payload = {
        "probe": "coordexp_swift_fa2_length_precision",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "device": str(device),
        "seed": args.seed,
        "dtype": resolved.config.training.precision,
        "attn_implementation": resolved.config.model.attn_implementation,
        "example_count": args.example_count,
        "target_index": args.target_index,
        "target_example_id": target.example_id,
        "positions_per_target": args.positions_per_target,
        "sweep_lengths": list(args.sweep_lengths),
        "variants": receipts,
        "comparisons": comparisons,
        "verdict": _verdict(comparisons, tolerance=args.max_abs_tolerance),
        "tolerance": {"max_abs": args.max_abs_tolerance},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "verdict": payload["verdict"]}))
    return 0


class PackVariant:
    def __init__(
        self,
        *,
        name: str,
        examples: tuple[Any, ...],
        pack: PackedSequence,
        target_example_id: str,
    ) -> None:
        self.name = name
        self.examples = examples
        self.pack = pack
        self.target_example_id = target_example_id

    def to_artifact_dict(self) -> dict[str, Any]:
        target = _target_segment(self.pack, self.target_example_id)
        return {
            "name": self.name,
            "pack_length": self.pack.length,
            "global_max_length": self.pack.global_max_length,
            "segment_count": len(self.pack.segments),
            "target_example_id": self.target_example_id,
            "target_start": target.start,
            "target_end": target.end,
            "target_length": target.length,
            "segments": [segment.to_artifact_dict() for segment in self.pack.segments],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/coordexp_swift/length_isolation/precision_probe"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--example-count", type=int, default=10)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument(
        "--sweep-lengths",
        type=int,
        nargs="+",
        default=(6_000, 12_000, 16_000, 20_000, 24_000, 32_000),
        help="Target physical pack lengths to approximate with isolated segments.",
    )
    parser.add_argument("--positions-per-target", type=int, default=256)
    parser.add_argument("--max-abs-tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


def _load_encoded_examples(
    config: Any,
    *,
    components: Any,
    example_count: int,
) -> tuple[Any, ...]:
    raw_examples = load_raw_examples(config.data.train, sample_limit=example_count)
    encoded = []
    for raw in raw_examples:
        rendered = render_example(raw, config.template)
        encoded.append(
            encode_rendered_example(
                raw,
                rendered,
                components=components,
                processor_config=config.model.processor,
                global_max_length=12_000,
                materialize_image_pixels=False,
            )
        )
    return tuple(encoded)


def _build_sweep_variants(
    encoded: tuple[Any, ...],
    *,
    target_index: int,
    target_lengths: tuple[int, ...],
) -> dict[str, PackVariant]:
    if target_index < 0 or target_index >= len(encoded):
        raise ValueError("target_index is outside encoded examples")
    target = encoded[target_index]
    filler = tuple(example for index, example in enumerate(encoded) if index != target_index)
    specs: dict[str, tuple[tuple[Any, ...], int]] = {
        "alone": ((target,), max(max(target_lengths), len(target.input_ids))),
    }
    for target_length in sorted(set(target_lengths)):
        first_examples = (target, *_fillers_until_limit(target, filler, target_length))
        last_fillers = _fillers_until_limit(target, filler, target_length)
        specs[f"first_{target_length}"] = (first_examples, target_length)
        specs[f"last_{target_length}"] = ((*last_fillers, target), target_length)
    largest = max(target_lengths)
    repeat_examples = (target, *_fillers_until_limit(target, filler, largest))
    specs[f"first_{largest}_repeat"] = (repeat_examples, largest)
    variants = {}
    for name, (examples, global_max_length) in specs.items():
        packs = plan_packed_sequences(examples, global_max_length=global_max_length)
        if len(packs) != 1:
            raise RuntimeError(
                f"variant {name} did not fit into one pack: {len(packs)} packs"
            )
        variants[name] = PackVariant(
            name=name,
            examples=tuple(examples),
            pack=packs[0],
            target_example_id=target.example_id,
        )
    return variants


def _fillers_until_limit(
    target: Any,
    fillers: tuple[Any, ...],
    global_max_length: int,
) -> tuple[Any, ...]:
    chosen: list[Any] = []
    total = len(target.input_ids)
    for filler in fillers:
        next_length = total + len(filler.input_ids)
        if next_length > global_max_length:
            continue
        chosen.append(filler)
        total = next_length
        if total >= global_max_length * 0.97:
            break
    return tuple(chosen)


def _build_comparisons(
    logits_by_variant: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    if "alone" not in logits_by_variant:
        raise RuntimeError("alone baseline is missing")
    for name, logits in sorted(logits_by_variant.items()):
        if name == "alone" or name.endswith("_repeat"):
            continue
        comparisons[f"alone_vs_{name}"] = _compare_logits(logits_by_variant["alone"], logits)
    for name, logits in sorted(logits_by_variant.items()):
        if not name.endswith("_repeat"):
            continue
        base_name = name.removesuffix("_repeat")
        if base_name in logits_by_variant:
            comparisons[f"repeat_{base_name}"] = _compare_logits(
                logits_by_variant[base_name],
                logits,
            )
    for name, logits in sorted(logits_by_variant.items()):
        if not name.startswith("last_"):
            continue
        length = name.removeprefix("last_")
        first_name = f"first_{length}"
        if first_name in logits_by_variant:
            comparisons[f"{first_name}_vs_{name}"] = _compare_logits(
                logits_by_variant[first_name],
                logits,
            )
    return comparisons


def _target_segment(pack: PackedSequence, target_example_id: str) -> Any:
    matches = [
        segment for segment in pack.segments if segment.example_id == target_example_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one target segment for {target_example_id}, found {len(matches)}"
        )
    return matches[0]


def _select_offsets(
    target_length: int,
    *,
    positions_per_target: int,
) -> tuple[int, ...]:
    if positions_per_target <= 0:
        raise ValueError("positions_per_target must be positive")
    if target_length <= positions_per_target:
        return tuple(range(target_length))
    values = {
        round(index * (target_length - 1) / (positions_per_target - 1))
        for index in range(positions_per_target)
    }
    return tuple(sorted(int(value) for value in values))


def _logits_2d(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise RuntimeError(f"expected batch size 1 logits, got {tuple(logits.shape)}")
        return logits[0]
    if logits.ndim == 2:
        return logits
    raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")


def _compare_logits(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise RuntimeError(f"logit shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}")
    bitwise_equal = torch.equal(left, right)
    left_f = left.float()
    right_f = right.float()
    diff = (left_f - right_f).abs()
    denom = torch.maximum(torch.maximum(left_f.abs(), right_f.abs()), torch.tensor(1.0e-6))
    rel = diff / denom
    left_top1 = left_f.argmax(dim=-1)
    right_top1 = right_f.argmax(dim=-1)
    return {
        "shape": list(left.shape),
        "bitwise_equal": bool(bitwise_equal),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p50_abs": _quantile(diff, 0.50),
        "p95_abs": _quantile(diff, 0.95),
        "p99_abs": _quantile(diff, 0.99),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
        "top1_mismatch_count": int((left_top1 != right_top1).sum().item()),
        "position_count": int(left.shape[0]),
        "vocab_size": int(left.shape[1]),
    }


def _quantile(values: torch.Tensor, q: float) -> float:
    flat = values.flatten()
    if flat.numel() == 0:
        return math.nan
    max_items = 2_000_000
    if flat.numel() > max_items:
        step = math.ceil(flat.numel() / max_items)
        flat = flat[::step]
    return float(torch.quantile(flat.contiguous(), q).item())


def _verdict(comparisons: dict[str, dict[str, Any]], *, tolerance: float) -> str:
    repeat = [
        value for key, value in comparisons.items() if key.startswith("repeat_")
    ]
    if not repeat:
        return "inconclusive_repeat_missing"
    cross_pack = [
        value for key, value in comparisons.items() if not key.startswith("repeat_")
    ]
    if any(item["max_abs"] > tolerance for item in repeat):
        return "inconclusive_repeat_not_stable"
    if all(item["max_abs"] <= tolerance and item["top1_mismatch_count"] == 0 for item in cross_pack):
        return "no_length_precision_instability_detected"
    return "length_or_offset_precision_delta_detected"


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CoordExp-Swift FA2 Length Precision Probe",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "This probe compares logits for the same target example positions across",
        "different packed-row layouts. It does not compare training loss curves,",
        "optimizer updates, or different numbers of examples per step.",
        "",
        "## Variants",
        "",
        "| variant | pack length | segments | target start | target length |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, record in sorted(payload["variants"].items()):
        variant = record["variant"]
        lines.append(
            "| {name} | {pack_length} | {segment_count} | {target_start} | {target_length} |".format(
                name=name,
                pack_length=variant["pack_length"],
                segment_count=variant["segment_count"],
                target_start=variant["target_start"],
                target_length=variant["target_length"],
            )
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| comparison | bitwise | max abs | mean abs | p99 abs | max rel | top1 mismatches |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, comparison in sorted(payload["comparisons"].items()):
        lines.append(
            "| {name} | {bitwise_equal} | {max_abs:.6g} | {mean_abs:.6g} | {p99_abs:.6g} | {max_rel:.6g} | {top1_mismatch_count} |".format(
                name=name,
                **comparison,
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Config: `{payload['config_path']}`",
            f"- Device: `{payload['device']}`",
            f"- Dtype: `{payload['dtype']}`",
            f"- Attention implementation: `{payload['attn_implementation']}`",
            f"- Selected positions per target: `{payload['positions_per_target']}`",
            f"- Max-abs tolerance: `{payload['tolerance']['max_abs']}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
