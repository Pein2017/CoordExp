#!/usr/bin/env python
"""Unified pipeline runner for CoordExp inference/eval/vis (YAML-first).

Primary usage:
  python scripts/run_infer.py --config configs/infer/<exp>.yaml

The YAML config is treated as a single file (no extends/inherit, no variable
interpolation). Legacy CLI flags are supported during a transition period:
- If both --config and legacy flags are provided, legacy flags override YAML.

For legacy (flag-only) inference runs (no YAML):
  python scripts/run_infer.py \
      --gt_jsonl <path> \
      --model_checkpoint <ckpt> \
      --mode coord|text|auto \
      --out <path/to/gt_vs_pred.jsonl>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.infer.pipeline import run_pipeline


def _add_legacy_infer_flags(ap: argparse.ArgumentParser, *, required: bool) -> None:
    ap.add_argument("--gt_jsonl", required=required, help="Path to ground-truth JSONL")
    ap.add_argument("--model_checkpoint", required=required, help="Checkpoint path")
    ap.add_argument(
        "--mode",
        required=required,
        choices=["coord", "text", "auto"],
        help="Model/GT mode (coord-token vs pixel GT), or auto-detect",
    )
    ap.add_argument(
        "--pred-coord-mode",
        choices=["auto", "pixel", "norm1000"],
        default=None if not required else "auto",
        help="Override how prediction coords are interpreted before scaling",
    )
    ap.add_argument("--device", default=None if not required else "cuda:0")
    ap.add_argument(
        "--limit", type=int, default=None if not required else 0, help="0 = all"
    )
    ap.add_argument(
        "--detect-samples",
        type=int,
        default=None if not required else 128,
        help="When mode=auto, how many GT records to scan",
    )

    # Artifacts (legacy-only; in YAML mode these map to artifacts.* overrides)
    ap.add_argument(
        "--out",
        default=None if not required else "gt_vs_pred.jsonl",
        help="Output JSONL path (defaults to gt_vs_pred.jsonl)",
    )
    ap.add_argument(
        "--summary",
        default=None,
        help="Optional summary path (defaults to <out_dir>/summary.json)",
    )

    # Backend
    ap.add_argument(
        "--backend",
        choices=["hf", "vllm"],
        default=None if not required else "hf",
        help="Generation backend: hf (default) or vllm",
    )
    ap.add_argument(
        "--vllm-base-url",
        default=None,
        help="(vllm backend) OpenAI-compatible base URL, e.g. http://127.0.0.1:8000",
    )
    ap.add_argument(
        "--vllm-model",
        default=None,
        help="(vllm backend) Served model name; defaults to --model_checkpoint",
    )

    # Generation flags
    ap.add_argument("--temperature", type=float, default=None if not required else 0.01)
    ap.add_argument("--top_p", type=float, default=None if not required else 0.95)
    ap.add_argument(
        "--max_new_tokens", type=int, default=None if not required else 1024
    )
    ap.add_argument(
        "--repetition_penalty", type=float, default=None if not required else 1.05
    )
    ap.add_argument("--seed", type=int, default=None)


def _yaml_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    o: Dict[str, Any] = {}

    def _set(k: str, v: Any) -> None:
        if v is not None:
            o[k] = v

    # Run/artifacts overrides
    _set("artifacts.gt_vs_pred_jsonl", args.out)
    _set("artifacts.summary_json", args.summary)

    # Infer overrides
    _set("infer.gt_jsonl", args.gt_jsonl)
    _set("infer.model_checkpoint", args.model_checkpoint)
    _set("infer.mode", args.mode)
    _set("infer.pred_coord_mode", args.pred_coord_mode)
    _set("infer.device", args.device)
    _set("infer.limit", args.limit)
    _set("infer.detect_samples", args.detect_samples)

    # Backend overrides
    _set("infer.backend.type", args.backend)
    _set("infer.backend.base_url", args.vllm_base_url)
    _set("infer.backend.model", args.vllm_model)

    # Generation overrides
    _set("infer.generation.temperature", args.temperature)
    _set("infer.generation.top_p", args.top_p)
    _set("infer.generation.max_new_tokens", args.max_new_tokens)
    _set("infer.generation.repetition_penalty", args.repetition_penalty)
    _set("infer.generation.seed", args.seed)

    return o


def _run_legacy_infer(args: argparse.Namespace) -> None:
    from src.infer import GenerationConfig, InferenceConfig, InferenceEngine

    gen_cfg = GenerationConfig(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=int(args.max_new_tokens),
        repetition_penalty=float(args.repetition_penalty),
        seed=args.seed,
    )

    backend_cfg: Dict[str, Any] = {}
    if args.backend == "vllm":
        if args.vllm_base_url:
            backend_cfg["base_url"] = str(args.vllm_base_url)
        if args.vllm_model:
            backend_cfg["model"] = str(args.vllm_model)

    inf_cfg = InferenceConfig(
        gt_jsonl=str(args.gt_jsonl),
        model_checkpoint=str(args.model_checkpoint),
        mode=str(args.mode),
        pred_coord_mode=str(args.pred_coord_mode),
        out_path=str(args.out),
        summary_path=str(args.summary) if args.summary else None,
        device=str(args.device),
        limit=int(args.limit),
        backend_type=str(args.backend),
        backend=backend_cfg,
        detect_samples=int(args.detect_samples),
    )

    engine = InferenceEngine(inf_cfg, gen_cfg)
    out_path, summary_path = engine.infer()
    print(f"Wrote predictions to {out_path} and summary to {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified inference pipeline runner")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config (YAML-first).",
    )

    # If --config is provided, legacy flags become optional overrides.
    _add_legacy_infer_flags(ap, required=False)

    args = ap.parse_args()

    if args.config is not None:
        overrides = _yaml_overrides_from_args(args)
        artifacts = run_pipeline(config_path=Path(args.config), overrides=overrides)
        print(f"Pipeline complete. run_dir={artifacts.run_dir}")
        return

    # Legacy flag-only mode: require the classic inputs.
    ap2 = argparse.ArgumentParser(description="Legacy inference (no YAML)")
    _add_legacy_infer_flags(ap2, required=True)
    args2 = ap2.parse_args()
    _run_legacy_infer(args2)


if __name__ == "__main__":
    main()
