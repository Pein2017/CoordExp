from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def build_eval_artifact_paths(*, run_dir: Path, eval_dir: Path) -> Dict[str, Path]:
    return {
        "gt_vs_pred_guarded_jsonl": run_dir / "gt_vs_pred_guarded.jsonl",
        "gt_vs_pred_scored_guarded_jsonl": run_dir / "gt_vs_pred_scored_guarded.jsonl",
        "metrics_json": eval_dir / "metrics.json",
        "metrics_guarded_json": eval_dir / "metrics_guarded.json",
        "duplicate_guard_report_json": eval_dir / "duplicate_guard_report.json",
    }


def resolve_infer_artifact_paths(
    *,
    cfg: Any,
    backend: str,
) -> tuple[Path, Path, Optional[Path]]:
    out_path = Path(str(cfg.out_path))
    summary_path = Path(str(cfg.summary_path or (out_path.parent / "summary.json")))
    trace_path_raw = str(cfg.pred_token_trace_path or "").strip()
    if trace_path_raw:
        trace_path: Optional[Path] = Path(trace_path_raw)
    elif backend == "hf":
        trace_path = out_path.parent / "pred_token_trace.jsonl"
    else:
        trace_path = None
    return out_path, summary_path, trace_path


def ensure_infer_artifact_dirs(
    *,
    out_path: Path,
    summary_path: Path,
    trace_path: Optional[Path],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)


def build_infer_resolved_meta(
    *,
    owner: Any,
    backend: str,
    batch_size: int,
    out_path: Path,
    summary_path: Path,
    trace_path: Optional[Path],
) -> Dict[str, Any]:
    resolved_meta = {
        "mode": owner.resolved_mode,
        "mode_resolution_reason": owner.mode_reason,
        "backend": backend,
        "model_checkpoint": owner.cfg.model_checkpoint,
        "gt_jsonl": owner.cfg.gt_jsonl,
        "pred_coord_mode": owner.cfg.pred_coord_mode,
        "bbox_format": owner.cfg.bbox_format,
        "prompt_variant": owner.prompt_variant,
        "object_field_order": owner.object_field_order,
        "object_ordering": owner.object_ordering,
        "device": owner.cfg.device,
        "limit": owner.cfg.limit,
        "generation": {
            "temperature": owner.gen_cfg.temperature,
            "top_p": owner.gen_cfg.top_p,
            "max_new_tokens": owner.gen_cfg.max_new_tokens,
            "repetition_penalty": owner.gen_cfg.repetition_penalty,
            "batch_size": batch_size,
            "seed": owner.gen_cfg.seed,
        },
        "artifacts": {
            "gt_vs_pred_jsonl": str(out_path),
            "pred_token_trace_jsonl": str(trace_path) if trace_path is not None else None,
            "summary_json": str(summary_path),
        },
    }
    if bool(getattr(owner.cfg, "distributed_enabled", False)):
        resolved_meta["distributed"] = {
            "enabled": True,
            "rank": int(getattr(owner.cfg, "rank", 0) or 0),
            "local_rank": int(getattr(owner.cfg, "local_rank", 0) or 0),
            "world_size": max(int(getattr(owner.cfg, "world_size", 1) or 1), 1),
            "merge_strategy": "ordinal_restore",
        }
    if backend == "vllm":
        public_fields = {
            "mode",
            "base_url",
            "model",
            "timeout_s",
            "client_concurrency",
        }
        resolved_meta["backend_cfg"] = {
            k: v
            for k, v in (owner.cfg.backend or {}).items()
            if str(k) in public_fields
        }
    return resolved_meta


def build_infer_summary_payload(
    *,
    owner: Any,
    counters: Any,
    backend: str,
    determinism: str,
    batch_size: int,
) -> Dict[str, Any]:
    summary_payload: Dict[str, Any] = {
        "mode": owner.resolved_mode,
        "determinism": determinism,
        **counters.to_summary(),
        "backend": {
            "type": backend,
            "model_checkpoint": owner.cfg.model_checkpoint,
        },
        "generation": {
            "temperature": owner.gen_cfg.temperature,
            "top_p": owner.gen_cfg.top_p,
            "max_new_tokens": owner.gen_cfg.max_new_tokens,
            "repetition_penalty": owner.gen_cfg.repetition_penalty,
            "batch_size": batch_size,
            "seed": owner.gen_cfg.seed,
        },
        "infer": {
            "gt_jsonl": owner.cfg.gt_jsonl,
            "pred_coord_mode": owner.cfg.pred_coord_mode,
            "bbox_format": owner.cfg.bbox_format,
            "prompt_variant": owner.prompt_variant,
            "object_field_order": owner.object_field_order,
            "object_ordering": owner.object_ordering,
            "device": owner.cfg.device,
            "limit": owner.cfg.limit,
        },
    }
    if bool(getattr(owner.cfg, "distributed_enabled", False)):
        summary_payload["distributed"] = {
            "enabled": True,
            "rank": int(getattr(owner.cfg, "rank", 0) or 0),
            "local_rank": int(getattr(owner.cfg, "local_rank", 0) or 0),
            "world_size": max(int(getattr(owner.cfg, "world_size", 1) or 1), 1),
            "merge_strategy": "ordinal_restore",
        }

    if owner.requested_mode == "auto":
        summary_payload["mode_resolution_reason"] = owner.mode_reason

    if backend == "hf":
        summary_payload["backend"]["attn_implementation_requested"] = (
            owner.attn_implementation_requested
        )
        summary_payload["backend"]["attn_implementation_selected"] = (
            owner.attn_implementation_selected
        )
    return summary_payload


def write_infer_summary(*, summary_path: Path, summary_payload: Dict[str, Any]) -> None:
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
