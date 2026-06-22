from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.common.detection_sequence import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)
from src.config.prompts import (
    build_dense_system_prompt,
    build_dense_user_prompt,
    get_template_prompt_hash,
)
from src.detection.data import (
    CoordinateTokenBox,
    DetectionMetadata,
    NormalizedDetectionObject,
    NormalizedDetectionSample,
    ObjectOrderingPlan,
)
from src.detection.evaluation import parse_compact_full_output_artifact
from src.detection.template import get_detection_template
from src.infer.runtime import (
    create_offline_engine,
    make_offline_generation_result,
    parse_detection_template_output_artifact,
    _resolve_runtime_detection_template_id,
    _semantic_template_id_from_sequence_format,
)
from src.infer.pipeline import (
    _parser_policy_for_score_provenance,
    _write_scored_artifact_provenance,
    load_resolved_config,
    run_pipeline,
)


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
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 16,
                "repetition_penalty": 1.0,
                "batch_size": 1,
            },
            "backend": {"type": "hf"},
        },
        "detection_template": {"id": "compact"},
    }


def _load_infer_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict[str, object],
) -> SimpleNamespace:
    captured: dict[str, object] = {}

    def _fake_run_offline_inference(*, inference_kwargs, generation_kwargs, logger=None):
        del logger
        captured["detection_template_id"] = inference_kwargs["detection_template_id"]
        captured["parser_mode"] = inference_kwargs.get("parser_mode")
        captured["generation_keys"] = set(generation_kwargs)
        out_path = Path(str(inference_kwargs["out_path"]))
        summary_path = Path(str(inference_kwargs["summary_path"]))
        out_path.write_text("", encoding="utf-8")
        summary_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            base_jsonl_path=out_path,
            summary_path=summary_path,
            processor=None,
        )

    monkeypatch.setattr(
        "src.infer.pipeline.run_offline_inference",
        _fake_run_offline_inference,
    )

    config_path = tmp_path / "pipeline.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    artifacts = run_pipeline(config_path=config_path)
    resolved = load_resolved_config(artifacts.run_dir / "resolved_config.json")
    return SimpleNamespace(raw=resolved, captured=captured)


def test_new_teacher_forcing_infer_uses_semantic_template_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_infer_config(tmp_path, monkeypatch, _base_pipeline_cfg(tmp_path))

    assert config.raw["detection_template"]["id"] == "compact"
    assert config.raw["infer"]["detection_template_id"] == "compact"
    assert config.raw["infer"]["parsing"]["mode"] == "marker_delimited_strict"
    assert config.captured["detection_template_id"] == "compact"
    assert "compact_grammar" not in config.raw["infer"]["generation"]
    assert "compact_grammar_enabled" not in config.captured["generation_keys"]


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("detection_sequence_format", "compact_full", "infer.detection_sequence_format is retired"),
        ("row_separator", "none", "infer.row_separator is retired"),
        ("compact_full_parse_mode", "legacy_compatible", "infer.compact_full_parse_mode is retired"),
        ("parsing", {"compact_full": {"mode": "legacy_compatible"}}, "infer.parsing.compact_full is retired"),
    ],
)
def test_retired_infer_template_knobs_are_rejected(
    tmp_path: Path,
    key: str,
    value: object,
    match: str,
) -> None:
    cfg = _base_pipeline_cfg(tmp_path)
    infer_cfg = cfg["infer"]
    assert isinstance(infer_cfg, dict)
    infer_cfg[key] = value

    config_path = tmp_path / "bad_template_knob.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        run_pipeline(config_path=config_path)


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


def test_semantic_compact_parse_artifact_uses_configured_field_order() -> None:
    geometry_first_output = (
        f"{BOX_START_TOKEN}<|coord_1|><|coord_2|><|coord_10|><|coord_20|>{BOX_END_TOKEN}"
        f"{OBJECT_REF_START_TOKEN}cat{OBJECT_REF_END_TOKEN}"
    )

    artifact = parse_detection_template_output_artifact(
        geometry_first_output,
        detection_template_id="compact_object_box_closed",
        object_field_order="geometry_first",
    )
    default_artifact = parse_detection_template_output_artifact(
        geometry_first_output,
        detection_template_id="compact_object_box_closed",
    )

    assert artifact["raw_output_json"]["objects"] == [
        {
            "desc": "cat",
            "bbox_2d": [
                "<|coord_1|>",
                "<|coord_2|>",
                "<|coord_10|>",
                "<|coord_20|>",
            ],
        }
    ]
    assert artifact["parse_error_code"] is None
    assert default_artifact["raw_output_json"] is None
    assert default_artifact["parse_error_code"] == "strict_template_mismatch"


def test_base_compact_parse_artifact_uses_configured_field_order() -> None:
    geometry_first_output = (
        f"{BOX_START_TOKEN}<|coord_1|><|coord_2|><|coord_10|><|coord_20|>"
        f"{OBJECT_REF_START_TOKEN}cat"
    )

    artifact = parse_detection_template_output_artifact(
        geometry_first_output,
        detection_template_id="compact",
        object_field_order="geometry_first",
    )
    default_artifact = parse_detection_template_output_artifact(
        geometry_first_output,
        detection_template_id="compact",
    )

    assert artifact["object_field_order"] == "geometry_first"
    assert artifact["object_separator"] == BOX_START_TOKEN
    assert artifact["raw_output_json"]["objects"] == [
        {
            "desc": "cat",
            "bbox_2d": [
                "<|coord_1|>",
                "<|coord_2|>",
                "<|coord_10|>",
                "<|coord_20|>",
            ],
        }
    ]
    assert artifact["parse_error_code"] is None
    assert default_artifact["raw_output_json"] is None
    assert default_artifact["parse_error_code"] == "strict_template_mismatch"


def test_runtime_detection_template_fallback_canonicalizes_sequence_format() -> None:
    assert _semantic_template_id_from_sequence_format("coordjson") == "stage1_json_pretty"
    assert _semantic_template_id_from_sequence_format("compact") == "compact"
    assert _semantic_template_id_from_sequence_format("compact_full") == "compact"
    assert (
        _resolve_runtime_detection_template_id(
            SimpleNamespace(detection_sequence_format="compact_full")
        )
        == "compact"
    )


def test_compact_prompt_uses_template_newline_semantics() -> None:
    system_prompt = build_dense_system_prompt(
        prompt_variant="coco_80",
        detection_template_id="compact_object_box_closed_lines",
    )
    user_prompt = build_dense_user_prompt(
        prompt_variant="coco_80",
        detection_template_id="compact_object_box_closed_lines",
    )

    assert "<|object_ref_end|>" in system_prompt
    assert "<|box_end|>\n" in system_prompt
    assert "single newline" in system_prompt.lower()
    assert "including the final row" in user_prompt.lower()
    assert (
        get_template_prompt_hash(
            prompt_variant="coco_80",
            detection_template_id="compact_object_box_closed_lines",
        )
        != get_template_prompt_hash(
            prompt_variant="coco_80",
            detection_template_id="compact_object_box_closed",
        )
    )


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


def test_infer_artifact_writer_records_detection_template_parse_policy(
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

    engine = create_offline_engine(
        inference_kwargs={
            "gt_jsonl": str(gt_jsonl),
            "model_checkpoint": "dummy",
            "mode": "coord",
            "detection_template_id": "compact",
            "pred_coord_mode": "auto",
            "out_path": str(out_path),
            "summary_path": str(summary_path),
            "root_image_dir": str(tmp_path),
        },
        generation_kwargs={},
    )
    monkeypatch.setattr(type(engine), "load_model", lambda self: None)
    monkeypatch.setattr(
        type(engine),
        "_generate_batch",
        lambda self, images: [
            make_offline_generation_result(text=raw_output) for _ in images
        ],
    )
    engine.infer()

    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["parse_mode"] == "strict_expected"
    assert row["serialization_policy"] == "compact"
    assert row["object_separator"] == "<|object_ref_start|>"
    assert row["terminal_token"] == "<|im_end|>"
    assert row["parse_error_code"] is None
    assert row["raw_ends_with_im_end"] is True
    assert row["detection_template_id"] == "compact"
    assert row["object_field_order"] == "desc_first"


def test_infer_artifact_writer_records_geometry_first_base_compact_policy(
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

    raw_output = (
        f"{BOX_START_TOKEN}<|coord_1|><|coord_2|><|coord_10|><|coord_20|>"
        f"{OBJECT_REF_START_TOKEN}cat<|im_end|><|endoftext|>"
    )
    out_path = tmp_path / "out" / "gt_vs_pred.jsonl"
    summary_path = tmp_path / "out" / "summary.json"

    engine = create_offline_engine(
        inference_kwargs={
            "gt_jsonl": str(gt_jsonl),
            "model_checkpoint": "dummy",
            "mode": "coord",
            "detection_template_id": "compact",
            "object_field_order": "geometry_first",
            "pred_coord_mode": "auto",
            "out_path": str(out_path),
            "summary_path": str(summary_path),
            "root_image_dir": str(tmp_path),
        },
        generation_kwargs={},
    )
    monkeypatch.setattr(type(engine), "load_model", lambda self: None)
    monkeypatch.setattr(
        type(engine),
        "_generate_batch",
        lambda self, images: [
            make_offline_generation_result(text=raw_output) for _ in images
        ],
    )
    engine.infer()

    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["parse_mode"] == "strict_expected"
    assert row["serialization_policy"] == "compact"
    assert row["object_separator"] == BOX_START_TOKEN
    assert row["terminal_token"] == "<|im_end|>"
    assert row["parse_error_code"] is None
    assert row["raw_ends_with_im_end"] is True
    assert row["detection_template_id"] == "compact"
    assert row["object_field_order"] == "geometry_first"
    assert row["raw_output_json"]["objects"] == [
        {
            "desc": "cat",
            "bbox_2d": [
                "<|coord_1|>",
                "<|coord_2|>",
                "<|coord_10|>",
                "<|coord_20|>",
            ],
        }
    ]


@pytest.mark.parametrize(
    "detection_template_id",
    [
        "compact",
        "compact_box_closed",
        "compact_object_box_closed",
        "compact_object_box_closed_lines",
    ],
)
def test_infer_artifact_writer_parses_each_compact_template_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    detection_template_id: str,
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

    sample = NormalizedDetectionSample(
        images=(img_path.name,),
        objects=(
            NormalizedDetectionObject(
                normalized_object_index=0,
                source_object_index=0,
                object_instance_id="img:ann:0",
                desc="cat",
                bbox_2d=CoordinateTokenBox(
                    "<|coord_1|>",
                    "<|coord_2|>",
                    "<|coord_10|>",
                    "<|coord_20|>",
                ),
                category_id=1,
                category_name="cat",
                coco_ann_id=1,
            ),
        ),
        width=100,
        height=100,
        image_id=1,
        file_name=img_path.name,
        metadata=DetectionMetadata(source="unit", split="test"),
        object_ordering=ObjectOrderingPlan.sorted().with_realized((0,)),
    )
    raw_output = (
        get_detection_template(detection_template_id).render_assistant(sample).text
        + "<|im_end|>"
    )
    out_path = tmp_path / "out" / "gt_vs_pred.jsonl"
    summary_path = tmp_path / "out" / "summary.json"

    engine = create_offline_engine(
        inference_kwargs={
            "gt_jsonl": str(gt_jsonl),
            "model_checkpoint": "dummy",
            "mode": "coord",
            "detection_template_id": detection_template_id,
            "pred_coord_mode": "auto",
            "out_path": str(out_path),
            "summary_path": str(summary_path),
            "root_image_dir": str(tmp_path),
        },
        generation_kwargs={},
    )
    monkeypatch.setattr(type(engine), "load_model", lambda self: None)
    monkeypatch.setattr(
        type(engine),
        "_generate_batch",
        lambda self, images: [
            make_offline_generation_result(text=raw_output) for _ in images
        ],
    )
    engine.infer()

    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["detection_template_id"] == detection_template_id
    assert row["parse_error_code"] is None
    assert row["pred"][0]["desc"] == "cat"


def test_score_provenance_binds_parser_policy_to_object_field_order(
    tmp_path: Path,
) -> None:
    cfg = {
        "detection_template": {"id": "compact"},
        "infer": {"object_field_order": "geometry_first"},
    }
    raw_path = tmp_path / "gt_vs_pred.jsonl"
    scored_path = tmp_path / "gt_vs_pred_scored.jsonl"
    raw_path.write_text('{"gt":[],"pred":[],"width":1,"height":1}\n', encoding="utf-8")
    scored_path.write_text(
        '{"gt":[],"pred":[],"width":1,"height":1,"score":1.0}\n',
        encoding="utf-8",
    )
    raw_path.with_suffix(raw_path.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:unit",
                "decode_policy_fingerprint": "decode:unit",
                "model_identity_fingerprint": "model:unit",
                "score_policy": "none",
                "detection_template": {"id": "compact"},
                "detection_template_id": "compact",
                "object_field_order": "geometry_first",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert (
        _parser_policy_for_score_provenance(cfg)
        == "compact:marker_delimited_strict:geometry_first"
    )
    assert _parser_policy_for_score_provenance(
        {
            "detection_template": {"id": "compact"},
            "infer": {"object_field_order": "desc_first"},
        }
    ) == "compact:marker_delimited_strict:desc_first"
    assert _write_scored_artifact_provenance(
        cfg=cfg,
        raw_path=raw_path,
        scored_path=scored_path,
        policy_name="unit_score",
        score_source="unit",
        aggregation_rule="unit",
        token_span_rule="none",
        constant_score_value=1.0,
    )

    sidecar = json.loads(
        scored_path.with_suffix(scored_path.suffix + ".provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["detection_template_id"] == "compact"
    assert sidecar["detection_template"] == {"id": "compact"}
    assert sidecar["object_field_order"] == "geometry_first"
    assert sidecar["parser_policy"] == "compact:marker_delimited_strict:geometry_first"


def test_score_provenance_rejects_raw_config_object_field_order_mismatch(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "gt_vs_pred.jsonl"
    scored_path = tmp_path / "gt_vs_pred_scored.jsonl"
    raw_path.write_text('{"gt":[],"pred":[],"width":1,"height":1}\n', encoding="utf-8")
    scored_path.write_text(
        '{"gt":[],"pred":[],"width":1,"height":1,"score":1.0}\n',
        encoding="utf-8",
    )
    raw_path.with_suffix(raw_path.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:unit",
                "decode_policy_fingerprint": "decode:unit",
                "model_identity_fingerprint": "model:unit",
                "score_policy": "none",
                "detection_template": {"id": "compact"},
                "detection_template_id": "compact",
                "object_field_order": "geometry_first",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert not _write_scored_artifact_provenance(
        cfg={
            "detection_template": {"id": "compact"},
            "infer": {"object_field_order": "desc_first"},
        },
        raw_path=raw_path,
        scored_path=scored_path,
        policy_name="unit_score",
        score_source="unit",
        aggregation_rule="unit",
        token_span_rule="none",
        constant_score_value=1.0,
    )
    assert not scored_path.with_suffix(scored_path.suffix + ".provenance.json").exists()


def test_score_provenance_removes_stale_sidecar_on_object_field_order_mismatch(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "gt_vs_pred.jsonl"
    scored_path = tmp_path / "gt_vs_pred_scored.jsonl"
    sidecar_path = scored_path.with_suffix(scored_path.suffix + ".provenance.json")
    raw_path.write_text('{"gt":[],"pred":[],"width":1,"height":1}\n', encoding="utf-8")
    scored_path.write_text(
        '{"gt":[],"pred":[],"width":1,"height":1,"score":1.0}\n',
        encoding="utf-8",
    )
    raw_path.with_suffix(raw_path.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:unit",
                "decode_policy_fingerprint": "decode:unit",
                "model_identity_fingerprint": "model:unit",
                "score_policy": "none",
                "detection_template": {"id": "compact"},
                "detection_template_id": "compact",
                "object_field_order": "geometry_first",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sidecar_path.write_text(
        json.dumps(
            {
                "prompt_policy_fingerprint": "prompt:stale",
                "decode_policy_fingerprint": "decode:stale",
                "model_identity_fingerprint": "model:stale",
                "score_policy_fingerprint": "score_policy:v1:" + "a" * 64,
                "metric_bearing": True,
                "artifact_path": str(scored_path),
                "detection_template": {"id": "compact"},
                "detection_template_id": "compact",
                "object_field_order": "desc_first",
                "parser_policy": "compact:marker_delimited_strict:desc_first",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert not _write_scored_artifact_provenance(
        cfg={
            "detection_template": {"id": "compact"},
            "infer": {"object_field_order": "desc_first"},
        },
        raw_path=raw_path,
        scored_path=scored_path,
        policy_name="unit_score",
        score_source="unit",
        aggregation_rule="unit",
        token_span_rule="none",
        constant_score_value=1.0,
    )
    assert not sidecar_path.exists()
