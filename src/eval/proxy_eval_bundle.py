from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from src.eval.detection import EvalOptions, evaluate_and_save
from src.eval.proxy_views import (
    DEFAULT_METADATA_NAMESPACE,
    materialize_proxy_eval_views,
    supported_proxy_views,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_REPO_ROOT = (
    REPO_ROOT.parent.parent if REPO_ROOT.parent.name == ".worktrees" else REPO_ROOT
)


@dataclass(frozen=True)
class ProxyEvalBundleArtifacts:
    run_dir: Path
    scored_jsonl: Path
    proxy_views_dir: Path
    output_root: Path
    summary_json: Path


@dataclass
class ProxyEvalBundleOptions:
    views: list[str] = field(default_factory=lambda: list(supported_proxy_views()))
    metadata_namespace: str = DEFAULT_METADATA_NAMESPACE
    eval_options: EvalOptions = field(default_factory=EvalOptions)


def _default_output_dir(output_root: Path, view: str) -> Path:
    return output_root / f"eval_{view}"


def _resolve_existing_input_path(path_raw: str) -> Path:
    raw = Path(str(path_raw))
    if raw.is_absolute():
        return raw.absolute()
    for root in (REPO_ROOT, COMMON_REPO_ROOT):
        candidate = (root / raw).absolute()
        if candidate.exists():
            return candidate
    return (COMMON_REPO_ROOT / raw).absolute()


def _resolve_artifacts(cfg: Mapping[str, Any]) -> ProxyEvalBundleArtifacts:
    run_dir = Path(str(cfg.get("run_dir") or ""))
    if not str(run_dir):
        raise ValueError("run_dir must be set")

    artifacts_cfg = cfg.get("artifacts") if isinstance(cfg.get("artifacts"), Mapping) else {}
    scored_jsonl = Path(
        str(artifacts_cfg.get("scored_jsonl") or (run_dir / "gt_vs_pred_scored.jsonl"))
    )
    proxy_views_dir = Path(
        str(artifacts_cfg.get("proxy_views_dir") or (run_dir / "proxy_eval_views"))
    )
    output_root = Path(
        str(artifacts_cfg.get("output_root") or run_dir)
    )
    summary_json = Path(
        str(artifacts_cfg.get("summary_json") or (run_dir / "proxy_eval_bundle_summary.json"))
    )
    return ProxyEvalBundleArtifacts(
        run_dir=run_dir,
        scored_jsonl=scored_jsonl,
        proxy_views_dir=proxy_views_dir,
        output_root=output_root,
        summary_json=summary_json,
    )


def options_from_config(cfg: Mapping[str, Any]) -> ProxyEvalBundleOptions:
    views = cfg.get("views") or list(supported_proxy_views())
    if not isinstance(views, list) or not all(isinstance(v, str) for v in views):
        raise ValueError("views must be a list of strings")

    eval_cfg = cfg.get("eval") if isinstance(cfg.get("eval"), Mapping) else {}
    eval_output_dir = eval_cfg.get("output_dir")
    if eval_output_dir is not None:
        # Per-view output dirs are derived from output_root; a fixed output_dir would collide.
        raise ValueError(
            "eval.output_dir is unsupported for proxy bundles; use artifacts.output_root instead"
        )

    eval_options = EvalOptions(
        metrics=str(eval_cfg.get("metrics", "both")),
        strict_parse=bool(eval_cfg.get("strict_parse", False)),
        use_segm=bool(eval_cfg.get("use_segm", True)),
        iou_thrs=eval_cfg.get("iou_thrs", None),
        f1ish_iou_thrs=[
            float(x) for x in (eval_cfg.get("f1ish_iou_thrs", [0.3, 0.5]) or [])
        ],
        f1ish_pred_scope=str(eval_cfg.get("f1ish_pred_scope", "annotated")),
        output_dir=Path("unused_proxy_bundle_output_dir"),
        overlay=bool(eval_cfg.get("overlay", False)),
        overlay_k=int(eval_cfg.get("overlay_k", 12)),
        num_workers=int(eval_cfg.get("num_workers", 0)),
        semantic_model=str(
            _resolve_existing_input_path(str(eval_cfg.get("semantic_model")))
            if str(eval_cfg.get("semantic_model", "")).strip().startswith("model_cache/")
            else eval_cfg.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
        ),
        semantic_threshold=float(eval_cfg.get("semantic_threshold", 0.6)),
        semantic_device=str(eval_cfg.get("semantic_device", "auto")),
        semantic_batch_size=int(eval_cfg.get("semantic_batch_size", 64)),
        lvis_max_dets=int(eval_cfg.get("lvis_max_dets", 300)),
    )
    return ProxyEvalBundleOptions(
        views=views,
        metadata_namespace=str(
            cfg.get("metadata_namespace", DEFAULT_METADATA_NAMESPACE)
        ),
        eval_options=eval_options,
    )


def run_proxy_eval_bundle(
    artifacts: ProxyEvalBundleArtifacts,
    *,
    options: ProxyEvalBundleOptions,
) -> Dict[str, Any]:
    if not artifacts.scored_jsonl.is_file():
        raise FileNotFoundError(
            f"scored eval artifact not found: {artifacts.scored_jsonl}"
        )

    view_summary = materialize_proxy_eval_views(
        artifacts.scored_jsonl,
        output_dir=artifacts.proxy_views_dir,
        views=options.views,
        metadata_namespace=options.metadata_namespace,
    )

    results: Dict[str, Any] = {}
    for view in options.views:
        pred_jsonl = Path(view_summary["outputs"][view])
        per_view_options = EvalOptions(
            metrics=options.eval_options.metrics,
            strict_parse=options.eval_options.strict_parse,
            use_segm=options.eval_options.use_segm,
            iou_thrs=options.eval_options.iou_thrs,
            f1ish_iou_thrs=list(options.eval_options.f1ish_iou_thrs),
            f1ish_pred_scope=options.eval_options.f1ish_pred_scope,
            output_dir=_default_output_dir(artifacts.output_root, view),
            overlay=options.eval_options.overlay,
            overlay_k=options.eval_options.overlay_k,
            num_workers=options.eval_options.num_workers,
            semantic_model=options.eval_options.semantic_model,
            semantic_threshold=options.eval_options.semantic_threshold,
            semantic_device=options.eval_options.semantic_device,
            semantic_batch_size=options.eval_options.semantic_batch_size,
            lvis_max_dets=options.eval_options.lvis_max_dets,
        )
        summary = evaluate_and_save(pred_jsonl, options=per_view_options)
        results[view] = {
            "pred_jsonl": str(pred_jsonl),
            "output_dir": str(per_view_options.output_dir),
            "metrics": summary.get("metrics", {}),
            "counters": summary.get("counters", {}),
        }

    bundle_summary = {
        "run_dir": str(artifacts.run_dir),
        "scored_jsonl": str(artifacts.scored_jsonl),
        "proxy_views_dir": str(artifacts.proxy_views_dir),
        "output_root": str(artifacts.output_root),
        "views": results,
        "materialize_summary": view_summary,
    }
    artifacts.summary_json.write_text(
        json.dumps(bundle_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    bundle_summary["summary_json"] = str(artifacts.summary_json)
    return bundle_summary


__all__ = [
    "ProxyEvalBundleArtifacts",
    "ProxyEvalBundleOptions",
    "options_from_config",
    "run_proxy_eval_bundle",
    "_resolve_artifacts",
]
