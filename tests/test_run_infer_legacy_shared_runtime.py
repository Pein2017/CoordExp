from __future__ import annotations

import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path


def _load_run_infer_with_fake_pipeline(monkeypatch, calls: list[dict]):
    fake_yaml = types.ModuleType("yaml")
    fake_yaml.safe_load = lambda _raw: {}
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)

    fake_requests = types.ModuleType("requests")
    fake_requests.RequestException = RuntimeError
    fake_requests.get = lambda *_args, **_kwargs: types.SimpleNamespace(status_code=500)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    fake_pipeline = types.ModuleType("src.infer.pipeline")

    def _fake_run_pipeline(*, config_path, overrides=None):
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        calls.append({"config": payload, "overrides": overrides})
        return types.SimpleNamespace(
            run_dir=Path(payload["artifacts"]["gt_vs_pred_jsonl"]).parent,
            gt_vs_pred_jsonl=Path(payload["artifacts"]["gt_vs_pred_jsonl"]),
            summary_json=Path(payload["artifacts"]["summary_json"]),
        )

    fake_pipeline.run_pipeline = _fake_run_pipeline
    monkeypatch.setitem(sys.modules, "src.infer.pipeline", fake_pipeline)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_infer.py"
    spec = importlib.util.spec_from_file_location("_run_infer_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_flag_only_infer_routes_through_pipeline(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []
    run_infer = _load_run_infer_with_fake_pipeline(monkeypatch, calls)
    out_path = tmp_path / "gt_vs_pred.jsonl"
    summary_path = tmp_path / "summary.json"

    run_infer._run_legacy_infer(
        Namespace(
            gt_jsonl=tmp_path / "gt.jsonl",
            model_checkpoint="ckpt",
            mode="coord",
            pred_coord_mode="auto",
            device="cpu",
            limit=7,
            detect_samples=3,
            out=out_path,
            summary=summary_path,
            backend="hf",
            vllm_base_url=None,
            vllm_model=None,
            temperature=0.01,
            top_p=0.95,
            max_new_tokens=64,
            repetition_penalty=1.05,
            batch_size=2,
            seed=123,
        )
    )

    assert len(calls) == 1
    cfg = calls[0]["config"]
    assert calls[0]["overrides"] is None
    assert cfg["stages"] == {"infer": True, "eval": False, "vis": False}
    assert cfg["artifacts"]["gt_vs_pred_jsonl"] == str(out_path)
    assert cfg["artifacts"]["summary_json"] == str(summary_path)
    assert cfg["infer"]["backend"] == {"type": "hf"}
    assert cfg["infer"]["generation"]["batch_size"] == 2
    assert cfg["infer"]["generation"]["seed"] == 123
