from __future__ import annotations
import json
from pathlib import Path

import pytest

from src.eval.proxy_eval_bundle import (
    ProxyEvalBundleArtifacts,
    options_from_config,
    run_proxy_eval_bundle,
    _resolve_artifacts,
)


def test_resolve_proxy_eval_bundle_artifacts_defaults(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cfg = {"run_dir": str(run_dir)}
    artifacts = _resolve_artifacts(cfg)

    assert artifacts.run_dir == run_dir
    assert artifacts.scored_jsonl == run_dir / "gt_vs_pred_scored.jsonl"
    assert artifacts.proxy_views_dir == run_dir / "proxy_eval_views"
    assert artifacts.output_root == run_dir
    assert artifacts.summary_json == run_dir / "proxy_eval_bundle_summary.json"


def test_options_from_config_reads_eval_settings() -> None:
    cfg = {
        "views": ["coco_real", "coco_real_strict"],
        "eval": {
            "metrics": "both",
            "use_segm": False,
            "strict_parse": True,
            "semantic_model": "model_cache/all-MiniLM-L6-v2-local",
            "semantic_threshold": 0.5,
            "semantic_device": "cuda:0",
            "semantic_batch_size": 64,
            "f1ish_iou_thrs": [0.3, 0.5],
            "f1ish_pred_scope": "annotated",
            "num_workers": 8,
        },
    }
    options = options_from_config(cfg)

    assert options.views == ["coco_real", "coco_real_strict"]
    assert options.eval_options.metrics == "both"
    assert options.eval_options.use_segm is False
    assert options.eval_options.strict_parse is True
    assert options.eval_options.semantic_model.endswith("model_cache/all-MiniLM-L6-v2-local")
    assert options.eval_options.semantic_threshold == 0.5
    assert options.eval_options.semantic_device == "cuda:0"
    assert options.eval_options.semantic_batch_size == 64


def test_options_from_config_resolves_local_model_cache_path(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    local_model = repo_root / "model_cache" / "all-MiniLM-L6-v2-local"
    local_model.mkdir(parents=True)
    monkeypatch.setattr("src.eval.proxy_eval_bundle.REPO_ROOT", repo_root)
    monkeypatch.setattr("src.eval.proxy_eval_bundle.COMMON_REPO_ROOT", repo_root)

    cfg = {
        "eval": {
            "semantic_model": "model_cache/all-MiniLM-L6-v2-local",
        },
    }

    options = options_from_config(cfg)
    assert options.eval_options.semantic_model == str(local_model)


def test_proxy_eval_bundle_requires_source_score_provenance_for_official_metrics(
    tmp_path: Path,
) -> None:
    artifacts = ProxyEvalBundleArtifacts(
        run_dir=tmp_path,
        scored_jsonl=tmp_path / "gt_vs_pred_scored.jsonl",
        proxy_views_dir=tmp_path / "proxy_views",
        output_root=tmp_path / "eval",
        summary_json=tmp_path / "proxy_summary.json",
    )
    artifacts.scored_jsonl.write_text(
        json.dumps(
            {
                "image": "demo.jpg",
                "width": 1,
                "height": 1,
                "gt": [],
                "pred": [{"bbox": [0, 0, 1, 1], "desc": "cat", "score": 1.0}],
                "pred_score_source": "test",
                "pred_score_version": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    options = options_from_config({"eval": {"metrics": "coco"}})

    with pytest.raises(ValueError, match="missing_provenance"):
        run_proxy_eval_bundle(artifacts, options=options)
