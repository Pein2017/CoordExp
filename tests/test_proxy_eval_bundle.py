from __future__ import annotations
from pathlib import Path

from src.eval.proxy_eval_bundle import _resolve_artifacts, options_from_config


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
