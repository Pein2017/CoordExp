#!/usr/bin/env python3
"""
Inspect a local Qwen3-VL checkpoint and print:
  - model layer counts (LLM num_hidden_layers, ViT depth)
  - detected aligner (deepstack) indices
  - a suggested PEFT `tuner.target_regex` for common DoRA/LoRA placement policies

Why this exists:
  - PEFT matches `target_modules` regexes with `re.fullmatch()` against *module names* (not parameter names).
  - A slightly-wrong regex (missing anchors / matching non-linear modules) can silently match nothing or crash.
  - This script lets you derive the correct module paths from the checkpoint itself to avoid repeated guesswork.

Usage (from repo root):
  conda run -n ms python scripts/tools/inspect_checkpoint_modules.py output/.../ckpt-3106

Examples:
  # Suggest a regex for: last 8 LLM blocks + aligner MLPs (freeze ViT in YAML)
  conda run -n ms python scripts/tools/inspect_checkpoint_modules.py output/.../ckpt-3106 \\
    --llm-last 8 --aligner-mlp --print-target-regex

  # Suggest a regex for: last 12 ViT blocks + last 8 LLM blocks + aligner MLPs
  conda run -n ms python scripts/tools/inspect_checkpoint_modules.py output/.../ckpt-3106 \\
    --vit-last 12 --llm-last 8 --aligner-mlp --print-target-regex
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set


@dataclass(frozen=True)
class CheckpointArch:
    llm_num_layers: Optional[int]
    vit_depth: Optional[int]
    deepstack_indices: List[int]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_weight_keys(ckpt_dir: Path) -> Iterable[str]:
    """Yield parameter keys from a HF checkpoint without loading tensors."""
    idx_candidates = [
        ckpt_dir / "model.safetensors.index.json",
        ckpt_dir / "pytorch_model.bin.index.json",
    ]
    for idx_path in idx_candidates:
        if idx_path.exists():
            idx = _read_json(idx_path)
            wm = idx.get("weight_map", {})
            if isinstance(wm, dict):
                yield from wm.keys()
                return

    # Single-file safetensors fallback.
    st_path = ckpt_dir / "model.safetensors"
    if st_path.exists():
        try:
            from safetensors import safe_open  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "safetensors is required to inspect single-file checkpoints. "
                "Run with: conda run -n ms python ..."
            ) from exc

        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            yield from f.keys()
        return

    raise FileNotFoundError(
        "Could not find a supported weight index in the checkpoint dir. "
        "Expected one of: model.safetensors.index.json, pytorch_model.bin.index.json, model.safetensors"
    )


def _module_names_from_weight_keys(weight_keys: Iterable[str]) -> Set[str]:
    modules: Set[str] = set()
    for k in weight_keys:
        if k.endswith((".weight", ".bias")):
            modules.add(k.rsplit(".", 1)[0])
    return modules


def _load_arch(ckpt_dir: Path, modules: Set[str]) -> CheckpointArch:
    cfg_path = ckpt_dir / "config.json"
    llm_num_layers: Optional[int] = None
    vit_depth: Optional[int] = None
    if cfg_path.exists():
        cfg = _read_json(cfg_path)
        text_cfg = cfg.get("text_config") or {}
        vision_cfg = cfg.get("vision_config") or {}
        if isinstance(text_cfg, dict):
            llm_num_layers = text_cfg.get("num_hidden_layers")
        if isinstance(vision_cfg, dict):
            vit_depth = vision_cfg.get("depth")

    deepstack_indices: Set[int] = set()
    for m in modules:
        mm = re.match(r"model\.visual\.deepstack_merger_list\.(\d+)\.", m)
        if mm is not None:
            deepstack_indices.add(int(mm.group(1)))

    return CheckpointArch(
        llm_num_layers=int(llm_num_layers) if llm_num_layers is not None else None,
        vit_depth=int(vit_depth) if vit_depth is not None else None,
        deepstack_indices=sorted(deepstack_indices),
    )


def _format_int_alternation(indices: Sequence[int]) -> str:
    # Keep it explicit and readable; indices are small (typical N=4/8/12).
    return "|".join(str(i) for i in indices)


def _build_target_regex(
    *,
    llm_layers: Optional[Sequence[int]],
    vit_layers: Optional[Sequence[int]],
    deepstack_indices: Sequence[int],
    include_aligner_mlp: bool,
) -> str:
    parts: List[str] = []

    if include_aligner_mlp:
        if deepstack_indices:
            deepstack_alt = _format_int_alternation(deepstack_indices)
        else:
            # Defensive default; user can verify via this same script.
            deepstack_alt = "0|1|2"
        parts.append(
            rf"model\.visual\.merger\.(linear_fc1|linear_fc2)"
            rf"|model\.visual\.deepstack_merger_list\.({deepstack_alt})\.(linear_fc1|linear_fc2)"
        )

    if vit_layers is not None and len(vit_layers) > 0:
        vit_alt = _format_int_alternation(vit_layers)
        parts.append(
            rf"model\.visual\.blocks\.({vit_alt})\."
            rf"(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
        )

    if llm_layers is not None and len(llm_layers) > 0:
        llm_alt = _format_int_alternation(llm_layers)
        parts.append(
            rf"model\.language_model\.layers\.({llm_alt})\."
            rf"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
        )

    if not parts:
        raise ValueError("No targets selected; pass --llm-last/--vit-last and/or --aligner-mlp.")

    return rf"^({'|'.join(parts)})$"


def _last_n(total: Optional[int], n: int) -> List[int]:
    if total is None:
        raise ValueError("Checkpoint config.json missing layer counts; cannot compute last-N indices.")
    if n <= 0:
        raise ValueError(f"N must be > 0, got {n}")
    start = max(0, total - n)
    return list(range(start, total))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str, help="Checkpoint directory (HF format).")
    parser.add_argument("--llm-last", type=int, default=None, help="Target the last N LLM blocks.")
    parser.add_argument("--vit-last", type=int, default=None, help="Target the last N ViT blocks.")
    parser.add_argument(
        "--aligner-mlp",
        action="store_true",
        help="Include the aligner MLPs (visual.merger + deepstack_merger_list.* linear_fc1/2).",
    )
    parser.add_argument(
        "--print-target-regex",
        action="store_true",
        help="Print a suggested PEFT target_regex and how many modules it matches in this checkpoint.",
    )
    args = parser.parse_args()

    ckpt_dir = Path(os.path.expanduser(args.ckpt_dir)).resolve()
    weight_keys = list(_iter_weight_keys(ckpt_dir))
    modules = _module_names_from_weight_keys(weight_keys)
    arch = _load_arch(ckpt_dir, modules)

    print(f"ckpt_dir: {ckpt_dir}")
    print(f"llm_num_layers: {arch.llm_num_layers}")
    print(f"vit_depth: {arch.vit_depth}")
    print(f"deepstack_merger_list indices: {arch.deepstack_indices}")
    print(f"num_modules (weight/bias parents): {len(modules)}")

    if not args.print_target_regex:
        # Print a tiny, useful preview.
        preview = sorted([m for m in modules if m.startswith("model.visual.merger")])[:20]
        if preview:
            print("\nexample modules (aligner):")
            for m in preview:
                print(f"  {m}")
        preview = sorted([m for m in modules if m.startswith("model.language_model.layers.")])[:20]
        if preview:
            print("\nexample modules (llm):")
            for m in preview:
                print(f"  {m}")
        preview = sorted([m for m in modules if m.startswith("model.visual.blocks.")])[:20]
        if preview:
            print("\nexample modules (vit):")
            for m in preview:
                print(f"  {m}")
        return

    llm_layers = _last_n(arch.llm_num_layers, args.llm_last) if args.llm_last is not None else None
    vit_layers = _last_n(arch.vit_depth, args.vit_last) if args.vit_last is not None else None

    target_regex = _build_target_regex(
        llm_layers=llm_layers,
        vit_layers=vit_layers,
        deepstack_indices=arch.deepstack_indices,
        include_aligner_mlp=bool(args.aligner_mlp),
    )

    pat = re.compile(target_regex)
    matched = [m for m in modules if pat.fullmatch(m)]

    print("\nSuggested tuner.target_regex:")
    print(target_regex)
    print(f"\nMatched modules in this checkpoint: {len(matched)}")


if __name__ == "__main__":
    main()
