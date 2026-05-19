from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.common.detection_sequence import BOX_START_TOKEN, OBJECT_REF_START_TOKEN
from src.detection.evaluation import parse_compact_full_output_artifact
from src.infer.engine import (
    GenerationConfig,
    GenerationResult,
    InferenceConfig,
    InferenceEngine,
)
from src.infer.pipeline import load_resolved_config, run_pipeline


def _compact_row(desc: str, x1: int, y1: int, x2: int, y2: int) -> str:
    return (
        f"{OBJECT_REF_START_TOKEN}{desc}{BOX_START_TOKEN}"
        f"<|coord_{x1}|><|coord_{y1}|><|coord_{x2}|><|coord_{y2}|>"
    )


def _base_pipeline_cfg(tmp_path: Path) -> dict[str, object]:
    gt_jsonl = tmp_path / "data" / "gt.jsonl"
    gt_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gt_jsonl.write_text("", encoding="utf-8")
    return {
        "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
        "stages": {"infer": True, "eval": False, "vis": False},
        "infer": {
            "gt_jsonl": str(gt_jsonl),
            "model_checkpoint": "dummy",
            "mode": "coord",
            "pred_coord_mode": "auto",
            "prompt_variant": "coco_80",
            "detection_sequence_format": "compact_full",
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 16,
                "repetition_penalty": 1.0,
                "batch_size": 1,
            },
            "backend": {"type": "hf"},
        },
    }


def _load_infer_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict[str, object],
) -> SimpleNamespace:
    import src.infer.engine as infer_engine

    captured: dict[str, object] = {}

    def _fake_infer(self):
        captured["compact_full_parse_mode"] = self.cfg.compact_full_parse_mode
        captured["compact_grammar_enabled"] = self.gen_cfg.compact_grammar_enabled
        Path(self.cfg.out_path).write_text("", encoding="utf-8")
        Path(self.cfg.summary_path or "").write_text("{}", encoding="utf-8")
        return Path(self.cfg.out_path), Path(self.cfg.summary_path or "")

    monkeypatch.setattr(infer_engine.InferenceEngine, "infer", _fake_infer)

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    artifacts = run_pipeline(config_path=config_path)
    resolved = load_resolved_config(artifacts.run_dir / "resolved_config.json")
    return SimpleNamespace(raw=resolved, captured=captured)


def test_new_teacher_forcing_infer_defaults_to_marker_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_infer_config(tmp_path, monkeypatch, _base_pipeline_cfg(tmp_path))

    assert config.raw["infer"]["parsing"]["compact_full"]["mode"] == (
        "marker_delimited_strict"
    )
    assert config.captured["compact_full_parse_mode"] == "marker_delimited_strict"
    assert (
        config.raw["infer"]["generation"]["compact_grammar"]["enabled"] is False
    )
    assert config.captured["compact_grammar_enabled"] is False


def test_legacy_parser_mode_requires_explicit_legacy_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_pipeline_cfg(tmp_path)
    infer_cfg = cfg["infer"]
    assert isinstance(infer_cfg, dict)
    infer_cfg["parsing"] = {"compact_full": {"mode": "legacy_compatible"}}

    config_path = tmp_path / "bad_legacy.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata.compatibility_namespace=legacy"):
        run_pipeline(config_path=config_path)

    cfg["metadata"] = {"compatibility_namespace": "legacy"}
    config = _load_infer_config(tmp_path, monkeypatch, cfg)

    assert config.raw["infer"]["parsing"]["compact_full"]["mode"] == (
        "legacy_compatible"
    )
    assert config.raw["metadata"]["compatibility_namespace"] == "legacy"


def test_parse_artifact_records_policy_and_separator() -> None:
    marker_output_with_two_objects = (
        _compact_row("cat", 1, 2, 10, 20)
        + _compact_row("dog", 30, 40, 80, 90)
        + "<|im_end|>"
    )

    artifact = parse_compact_full_output_artifact(
        marker_output_with_two_objects,
        parse_mode="marker_delimited_strict",
    )

    assert artifact["parse_mode"] == "marker_delimited_strict"
    assert artifact["serialization_policy"] == "marker_delimited"
    assert artifact["object_separator"] == "<|object_ref_start|>"
    assert artifact["terminal_token"] == "<|im_end|>"
    assert artifact["parse_error_code"] is None
    assert artifact["raw_output_json"]["objects"] == [
        {
            "desc": "cat",
            "bbox_2d": [
                "<|coord_1|>",
                "<|coord_2|>",
                "<|coord_10|>",
                "<|coord_20|>",
            ],
        },
        {
            "desc": "dog",
            "bbox_2d": [
                "<|coord_30|>",
                "<|coord_40|>",
                "<|coord_80|>",
                "<|coord_90|>",
            ],
        },
    ]


def test_parse_artifact_records_concrete_failure_code() -> None:
    legacy_newline_output = _compact_row("cat", 1, 2, 10, 20) + "\n"

    artifact = parse_compact_full_output_artifact(
        legacy_newline_output,
        parse_mode="marker_delimited_strict",
    )

    assert artifact["raw_output_json"] is None
    assert artifact["parse_mode"] == "marker_delimited_strict"
    assert artifact["parse_error_code"] == "legacy_separator_in_new_format"
    assert artifact["parse_error_offset"] == len(_compact_row("cat", 1, 2, 10, 20))


def test_infer_artifact_writer_records_compact_full_parse_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img_path = tmp_path / "img.png"
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(img_path)

    gt_jsonl = tmp_path / "gt.jsonl"
    gt_jsonl.write_text(
        json.dumps(
            {
                "images": [img_path.name],
                "width": 100,
                "height": 100,
                "objects": [
                    {
                        "desc": "cat",
                        "bbox_2d": [
                            "<|coord_1|>",
                            "<|coord_2|>",
                            "<|coord_10|>",
                            "<|coord_20|>",
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    raw_output = _compact_row("cat", 1, 2, 10, 20) + "<|im_end|><|endoftext|>"
    out_path = tmp_path / "out" / "gt_vs_pred.jsonl"
    summary_path = tmp_path / "out" / "summary.json"

    monkeypatch.setattr(InferenceEngine, "load_model", lambda self: None)
    monkeypatch.setattr(
        InferenceEngine,
        "_generate_batch",
        lambda self, images: [GenerationResult(text=raw_output) for _ in images],
    )

    engine = InferenceEngine(
        InferenceConfig(
            gt_jsonl=str(gt_jsonl),
            model_checkpoint="dummy",
            mode="coord",
            detection_sequence_format="compact_full",
            pred_coord_mode="auto",
            out_path=str(out_path),
            summary_path=str(summary_path),
            root_image_dir=str(tmp_path),
        ),
        GenerationConfig(),
    )
    engine.infer()

    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["parse_mode"] == "marker_delimited_strict"
    assert row["serialization_policy"] == "marker_delimited"
    assert row["object_separator"] == "<|object_ref_start|>"
    assert row["terminal_token"] == "<|im_end|>"
    assert row["parse_error_code"] is None
    assert row["raw_ends_with_im_end"] is True
