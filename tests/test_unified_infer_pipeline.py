from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from src.infer.engine import detect_mode_from_gt
from src.infer.pipeline import (
    load_resolved_config,
    resolve_artifacts,
    resolve_root_image_dir_for_jsonl,
    run_pipeline,
)
from src.infer.vis import render_vis_from_jsonl


def _write_jsonl(path: Path, records: list[object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            if isinstance(r, str):
                f.write(r + "\n")
            else:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_detect_mode_from_gt_coord_tokens(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "width": 100,
                "height": 100,
                "objects": [
                    {
                        "bbox_2d": [
                            "<|coord_10|>",
                            "<|coord_10|>",
                            "<|coord_20|>",
                            "<|coord_20|>",
                        ]
                    }
                ],
            }
        ],
    )

    mode, reason = detect_mode_from_gt(str(gt), sample_size=128)
    assert mode == "coord"
    assert reason == "coord_tokens_found"


def test_detect_mode_from_gt_points_exceed_image(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "width": 10,
                "height": 10,
                "objects": [{"bbox_2d": [0, 0, 999, 1]}],
            }
        ],
    )

    mode, reason = detect_mode_from_gt(str(gt), sample_size=128)
    assert mode == "coord"
    assert reason == "points_exceed_image"


def test_detect_mode_from_gt_within_bounds(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "width": 10,
                "height": 10,
                "objects": [{"bbox_2d": [0, 0, 9, 9]}],
            }
        ],
    )

    mode, reason = detect_mode_from_gt(str(gt), sample_size=128)
    assert mode == "text"
    assert reason == "within_image_bounds"


def test_detect_mode_from_gt_no_valid_records(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    _write_jsonl(
        gt,
        [
            {"width": 32, "height": 32, "objects": []},
            {"width": 32, "height": 32, "gt": []},
        ],
    )

    mode, reason = detect_mode_from_gt(str(gt), sample_size=128)
    assert mode == "text"
    assert reason == "no_valid_records"


def test_detect_mode_from_gt_rejects_malformed_jsonl(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    _write_jsonl(gt, ["not json"])

    with pytest.raises(ValueError, match="Malformed JSONL"):
        detect_mode_from_gt(str(gt), sample_size=128)


@pytest.mark.parametrize(
    ("record", "error_pattern"),
    [
        (
            {"height": 32, "objects": [{"bbox_2d": [0, 0, 10, 10]}]},
            "Missing width/height",
        ),
        (
            {"width": "abc", "height": 32, "objects": [{"bbox_2d": [0, 0, 10, 10]}]},
            "Invalid width/height",
        ),
    ],
)
def test_detect_mode_from_gt_rejects_invalid_size_records(
    tmp_path: Path,
    record: dict[str, object],
    error_pattern: str,
) -> None:
    gt = tmp_path / "gt.jsonl"
    _write_jsonl(gt, [record])

    with pytest.raises(ValueError, match=error_pattern):
        detect_mode_from_gt(str(gt), sample_size=128)


def test_pipeline_resolve_artifacts_defaults(tmp_path: Path) -> None:
    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "stages": {"infer": True, "eval": True, "vis": True},
        "infer": {
            "gt_jsonl": "x.jsonl",
            "model_checkpoint": "ckpt",
            "mode": "coord",
        },
    }
    artifacts, stages = resolve_artifacts(cfg)
    assert stages.infer and stages.eval and stages.vis
    assert artifacts.run_dir == tmp_path / "out" / "demo"
    assert artifacts.gt_vs_pred_jsonl == artifacts.run_dir / "gt_vs_pred.jsonl"
    assert artifacts.pred_token_trace_jsonl == artifacts.run_dir / "pred_token_trace.jsonl"
    assert artifacts.gt_vs_pred_scored_jsonl is None
    assert artifacts.summary_json == artifacts.run_dir / "summary.json"
    assert artifacts.eval_dir == artifacts.run_dir / "eval"
    assert artifacts.vis_dir == artifacts.run_dir / "vis"


def test_load_resolved_config_rejects_bad_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "resolved_config.json"
    path.write_text(json.dumps({"schema_version": "1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_resolved_config(path)

    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_resolved_config(path)


def test_load_resolved_config_treats_cfg_snapshot_as_opaque(tmp_path: Path) -> None:
    path = tmp_path / "resolved_config.json"
    payload = {
        "schema_version": 1,
        "config_path": "cfg.yaml",
        "root_image_dir": None,
        "root_image_dir_source": "none",
        "stages": {"infer": True, "eval": False, "vis": False},
        "artifacts": {"run_dir": "out/run"},
        "cfg": {
            "arbitrary": ["this", "shape", {"can": "change"}],
            "nested": {"non_contract": True},
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    loaded = load_resolved_config(path)
    assert loaded["schema_version"] == 1
    assert loaded["root_image_dir_source"] == "none"


def test_run_pipeline_writes_resolved_config_with_root_breadcrumbs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    gt_jsonl = tmp_path / "data" / "gt.jsonl"

    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "stages": {"infer": False, "eval": False, "vis": False},
        "infer": {"gt_jsonl": str(gt_jsonl)},
    }

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    artifacts, _stages = resolve_artifacts(cfg)
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts.gt_vs_pred_jsonl.write_text("", encoding="utf-8")

    run_pipeline(config_path=config_path)

    resolved_path = artifacts.run_dir / "resolved_config.json"
    resolved = load_resolved_config(resolved_path)
    assert resolved["schema_version"] == 1
    assert resolved["root_image_dir_source"] == "gt_parent"
    assert resolved["root_image_dir"] == str(gt_jsonl.parent.resolve())
    assert (
        resolved["artifacts"]["pred_token_trace_jsonl"]
        == str(artifacts.run_dir / "pred_token_trace.jsonl")
    )


def test_run_pipeline_writes_manifest_pointer_for_non_default_artifact_layout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    gt_jsonl = tmp_path / "data" / "gt.jsonl"
    external_jsonl = tmp_path / "external" / "preds" / "gt_vs_pred.jsonl"

    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "artifacts": {"gt_vs_pred_jsonl": str(external_jsonl)},
        "stages": {"infer": False, "eval": False, "vis": False},
        "infer": {"gt_jsonl": str(gt_jsonl)},
    }

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    external_jsonl.parent.mkdir(parents=True, exist_ok=True)
    external_jsonl.write_text("", encoding="utf-8")

    artifacts, _stages = resolve_artifacts(cfg)
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)

    run_pipeline(config_path=config_path)

    pointer_path = external_jsonl.parent / "resolved_config.path"
    assert pointer_path.exists()
    pointed = Path(pointer_path.read_text(encoding="utf-8").strip())
    assert pointed == (artifacts.run_dir / "resolved_config.json").resolve()


def test_resolve_root_image_dir_for_jsonl_uses_manifest_pointer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    jsonl_path = tmp_path / "external" / "gt_vs_pred.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("", encoding="utf-8")

    run_dir = tmp_path / "out" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    root_dir = tmp_path / "images"
    root_dir.mkdir(parents=True, exist_ok=True)

    resolved_payload = {
        "schema_version": 1,
        "config_path": "pipeline.yaml",
        "root_image_dir": str(root_dir),
        "root_image_dir_source": "config",
        "stages": {"infer": True, "eval": True, "vis": False},
        "artifacts": {
            "run_dir": str(run_dir),
            "gt_vs_pred_jsonl": str(jsonl_path),
            "summary_json": str(run_dir / "summary.json"),
            "eval_dir": str(run_dir / "eval"),
            "vis_dir": str(run_dir / "vis"),
        },
        "cfg": {},
    }
    resolved_path = run_dir / "resolved_config.json"
    resolved_path.write_text(json.dumps(resolved_payload, ensure_ascii=False), encoding="utf-8")

    (jsonl_path.parent / "resolved_config.path").write_text(
        str(resolved_path.resolve()),
        encoding="utf-8",
    )

    root, source = resolve_root_image_dir_for_jsonl(jsonl_path)
    assert root == root_dir.resolve()
    assert source == "config"


@pytest.mark.parametrize(
    "deprecated_key, value",
    [
        ("unknown_policy", "drop"),
        ("semantic_fallback", "drop"),
    ],
)
def test_pipeline_eval_stage_rejects_deprecated_eval_keys(
    tmp_path: Path,
    monkeypatch,
    deprecated_key: str,
    value: str,
) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    gt_jsonl = tmp_path / "data" / "gt.jsonl"
    gt_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gt_jsonl.write_text("", encoding="utf-8")

    external_jsonl = tmp_path / "external" / "preds" / "gt_vs_pred.jsonl"
    external_jsonl.parent.mkdir(parents=True, exist_ok=True)
    external_jsonl.write_text("", encoding="utf-8")

    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "artifacts": {"gt_vs_pred_jsonl": str(external_jsonl)},
        "stages": {"infer": False, "eval": True, "vis": False},
        "infer": {"gt_jsonl": str(gt_jsonl)},
        "eval": {deprecated_key: value},
    }

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    artifacts, _stages = resolve_artifacts(cfg)
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match=fr"eval\.{deprecated_key}"):
        run_pipeline(config_path=config_path)


def test_pipeline_eval_stage_rejects_legacy_use_pred_score_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)
    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    gt_jsonl = tmp_path / "data" / "gt.jsonl"
    gt_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gt_jsonl.write_text("", encoding="utf-8")

    base_jsonl = tmp_path / "external" / "gt_vs_pred.jsonl"
    base_jsonl.parent.mkdir(parents=True, exist_ok=True)
    base_jsonl.write_text("", encoding="utf-8")

    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "artifacts": {"gt_vs_pred_jsonl": str(base_jsonl)},
        "stages": {"infer": False, "eval": True, "vis": False},
        "infer": {"gt_jsonl": str(gt_jsonl)},
        "eval": {"metrics": "f1ish", "use_pred_score": True},
    }
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="eval.use_pred_score"):
        run_pipeline(config_path=config_path)


def test_pipeline_eval_stage_requires_scored_artifact_for_coco(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)
    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    gt_jsonl = tmp_path / "data" / "gt.jsonl"
    gt_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gt_jsonl.write_text("", encoding="utf-8")

    base_jsonl = tmp_path / "external" / "gt_vs_pred.jsonl"
    base_jsonl.parent.mkdir(parents=True, exist_ok=True)
    base_jsonl.write_text("", encoding="utf-8")

    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "artifacts": {"gt_vs_pred_jsonl": str(base_jsonl)},
        "stages": {"infer": False, "eval": True, "vis": False},
        "infer": {"gt_jsonl": str(gt_jsonl)},
        "eval": {"metrics": "coco"},
    }
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="artifacts.gt_vs_pred_scored_jsonl"):
        run_pipeline(config_path=config_path)


def test_pipeline_eval_stage_f1ish_uses_base_artifact_without_scored_jsonl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)
    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    gt_jsonl = tmp_path / "data" / "gt.jsonl"
    gt_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gt_jsonl.write_text("", encoding="utf-8")

    base_jsonl = tmp_path / "external" / "gt_vs_pred.jsonl"
    base_jsonl.parent.mkdir(parents=True, exist_ok=True)
    base_jsonl.write_text("", encoding="utf-8")

    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "artifacts": {"gt_vs_pred_jsonl": str(base_jsonl)},
        "stages": {"infer": False, "eval": True, "vis": False},
        "infer": {"gt_jsonl": str(gt_jsonl)},
        "eval": {"metrics": "f1ish"},
    }
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    import src.eval.detection as detection

    captured: dict[str, Path] = {}

    def _fake_eval(pred_path: Path, *, options):  # type: ignore[no-untyped-def]
        captured["pred_path"] = Path(pred_path)
        return {"metrics": {}, "per_class": {}, "counters": {}, "categories": {}}

    monkeypatch.setattr(detection, "evaluate_and_save", _fake_eval)

    run_pipeline(config_path=config_path)
    assert captured["pred_path"] == base_jsonl


def test_run_pipeline_passes_resolved_root_to_infer_without_env_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    config_root = tmp_path / "images_root"
    config_root.mkdir(parents=True, exist_ok=True)

    gt_jsonl = tmp_path / "data" / "gt.jsonl"
    gt_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gt_jsonl.write_text("", encoding="utf-8")

    cfg = {
        "run": {
            "name": "demo",
            "output_dir": str(tmp_path / "out"),
            "root_image_dir": str(config_root),
        },
        "stages": {"infer": True, "eval": False, "vis": False},
        "infer": {
            "gt_jsonl": str(gt_jsonl),
            "model_checkpoint": "dummy",
            "mode": "text",
            "pred_coord_mode": "auto",
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 4,
                "repetition_penalty": 1.0,
                "batch_size": 1,
            },
            "backend": {"type": "hf"},
        },
    }

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    import src.infer.engine as infer_engine

    captured: dict[str, object] = {}

    def _fake_infer(self):
        captured["root_image_dir"] = self.cfg.root_image_dir
        captured["pred_token_trace_path"] = self.cfg.pred_token_trace_path
        Path(self.cfg.out_path).write_text("", encoding="utf-8")
        Path(self.cfg.summary_path or "").write_text("{}", encoding="utf-8")
        return Path(self.cfg.out_path), Path(self.cfg.summary_path or "")

    monkeypatch.setattr(infer_engine.InferenceEngine, "infer", _fake_infer)

    run_pipeline(config_path=config_path)

    assert "ROOT_IMAGE_DIR" not in os.environ
    assert captured["root_image_dir"] == str(config_root.resolve())
    assert captured["pred_token_trace_path"] == str(
        (tmp_path / "out" / "demo" / "pred_token_trace.jsonl").resolve()
    )


def test_run_pipeline_wires_and_records_prompt_variant(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    gt_jsonl = tmp_path / "data" / "gt.jsonl"
    gt_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gt_jsonl.write_text("", encoding="utf-8")

    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "stages": {"infer": True, "eval": False, "vis": False},
        "infer": {
            "gt_jsonl": str(gt_jsonl),
            "model_checkpoint": "dummy",
            "mode": "text",
            "prompt_variant": "coco_80",
            "object_field_order": "geometry_first",
            "pred_coord_mode": "auto",
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 4,
                "repetition_penalty": 1.0,
                "batch_size": 1,
            },
            "backend": {"type": "hf"},
        },
    }

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    import src.infer.engine as infer_engine

    captured: dict[str, object] = {}

    def _fake_infer(self):
        captured["prompt_variant"] = self.cfg.prompt_variant
        captured["object_field_order"] = self.cfg.object_field_order
        Path(self.cfg.out_path).write_text("", encoding="utf-8")
        Path(self.cfg.summary_path or "").write_text("{}", encoding="utf-8")
        return Path(self.cfg.out_path), Path(self.cfg.summary_path or "")

    monkeypatch.setattr(infer_engine.InferenceEngine, "infer", _fake_infer)

    artifacts = run_pipeline(config_path=config_path)

    resolved = load_resolved_config(artifacts.run_dir / "resolved_config.json")
    assert captured["prompt_variant"] == "coco_80"
    assert captured["object_field_order"] == "geometry_first"
    assert resolved["infer"]["prompt_variant"] == "coco_80"
    assert resolved["infer"]["object_field_order"] == "geometry_first"


def test_run_pipeline_rejects_unknown_prompt_variant_with_available_keys(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "stages": {"infer": False, "eval": False, "vis": False},
        "infer": {
            "gt_jsonl": "unused.jsonl",
            "prompt_variant": "unknown_variant",
        },
    }

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown prompt variant") as exc_info:
        run_pipeline(config_path=config_path)

    message = str(exc_info.value)
    assert "unknown_variant" in message
    assert "default" in message
    assert "coco_80" in message


def test_run_pipeline_rejects_unknown_object_field_order(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "stages": {"infer": False, "eval": False, "vis": False},
        "infer": {
            "gt_jsonl": "unused.jsonl",
            "object_field_order": "invalid_order",
        },
    }

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="infer.object_field_order") as exc_info:
        run_pipeline(config_path=config_path)

    message = str(exc_info.value)
    assert "invalid_order" in message
    assert "desc_first" in message
    assert "geometry_first" in message

def test_pipeline_vis_stage_renders_using_resolved_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    yaml_stub = types.SimpleNamespace(safe_load=lambda raw: json.loads(raw))
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    img_path = data_dir / "img.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    cfg = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "stages": {"infer": False, "eval": False, "vis": True},
        "infer": {"gt_jsonl": str(data_dir / "gt.jsonl")},
        "vis": {"limit": 1},
    }

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    artifacts, _stages = resolve_artifacts(cfg)
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)

    artifacts.gt_vs_pred_jsonl.write_text(
        json.dumps(
            {
                "image": "img.png",
                "width": 64,
                "height": 64,
                "mode": "text",
                "coord_mode": "pixel",
                "gt": [{"type": "bbox_2d", "points": [1, 1, 10, 10], "desc": "a", "score": 1.0}],
                "pred": [{"type": "bbox_2d", "points": [2, 2, 11, 11], "desc": "a", "score": 1.0}],
                "raw_output_json": {},
                "raw_special_tokens": [],
                "raw_ends_with_im_end": True,
                "errors": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    run_pipeline(config_path=config_path)

    assert (artifacts.vis_dir / "vis_0000.png").exists()

    resolved = load_resolved_config(artifacts.run_dir / "resolved_config.json")
    assert resolved["root_image_dir"] == str(data_dir.resolve())


def test_vis_only_renders_without_model(tmp_path: Path, monkeypatch) -> None:
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    # Unified artifact schema expects `image` (not `images`).
    artifact = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(
        artifact,
        [
            {
                "image": "img.png",
                "width": 64,
                "height": 64,
                "mode": "text",
                "coord_mode": "pixel",
                "gt": [{"type": "bbox_2d", "points": [1, 1, 10, 10], "desc": "a", "score": 1.0}],
                "pred": [{"type": "bbox_2d", "points": [2, 2, 11, 11], "desc": "a", "score": 1.0}],
                "raw_output": "",
                "errors": [],
            }
        ],
    )

    monkeypatch.setenv("ROOT_IMAGE_DIR", str(tmp_path))

    out_dir = tmp_path / "vis"
    render_vis_from_jsonl(artifact, out_dir=out_dir, limit=1)

    assert (out_dir / "vis_0000.png").exists()
