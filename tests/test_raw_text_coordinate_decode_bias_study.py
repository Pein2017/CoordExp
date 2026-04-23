from __future__ import annotations

import json
from pathlib import Path
import types

import src.analysis.raw_text_coordinate_decode_bias_study as study_module
import yaml
from src.analysis.raw_text_coordinate_decode_bias_study import (
    hydrate_case_rows,
    load_study_config,
    run_study,
)

_BASE_MODEL_PATH = (
    "/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
)
_ADAPTER_PATH = (
    "/data/CoordExp/output/stage1_2b/"
    "coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/"
    "epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/"
    "v1-20260417-084341/checkpoint-552"
)


def _write_source_jsonl(path: Path) -> None:
    rows = [
        {
            "image_id": 101,
            "width": 100,
            "height": 100,
            "objects": [
                {"desc": "cup", "bbox_2d": [10, 20, 30, 40]},
                {"desc": "plate", "bbox_2d": [50, 60, 70, 80]},
            ],
        },
        {
            "image_id": 202,
            "width": 200,
            "height": 100,
            "objects": [
                {"desc": "book", "bbox_2d": [20, 10, 80, 50]},
                {"desc": "lamp", "bbox_2d": [100, 5, 150, 75]},
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_decode_source_jsonl(path: Path) -> None:
    rows = [
        {
            "image_id": 101,
            "width": 100,
            "height": 100,
            "images": ["images/000000000101.jpg"],
            "metadata": {
                "source": "coco",
                "split": "val",
                "coordexp_proxy_supervision": {
                    "object_supervision": "mixed",
                    "summary": {"strict_count": 1},
                },
            },
            "objects": [
                {"desc": "cup", "bbox_2d": [100, 200, 300, 400]},
                {"desc": "plate", "bbox_2d": [500, 600, 700, 800]},
            ],
        },
        {
            "image_id": 202,
            "width": 200,
            "height": 100,
            "images": ["images/000000000202.jpg"],
            "metadata": {
                "source": "coco",
                "split": "val",
                "coordexp_proxy_supervision": {
                    "object_supervision": "mixed",
                    "summary": {"strict_count": 2},
                },
            },
            "objects": [
                {"desc": "book", "bbox_2d": [100, 100, 400, 500]},
                {"desc": "lamp", "bbox_2d": [500, 50, 750, 750]},
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_config(
    *,
    config_path: Path,
    output_dir: Path,
    source_jsonl: Path,
    decode_source_jsonl: Path | None = None,
    stages: list[str] | None = None,
    stop_pressure_mode: str | None = None,
    stop_pressure_logit_bias: float = 0.0,
    include_adapter: bool = True,
) -> None:
    selected_stages = stages or [
        "hydrate",
        "counterfactual_eos",
        "counterfactual_repeat_penalty",
        "report",
    ]
    adapter_block = (
        f"""
  base_plus_adapter:
    alias: base_plus_adapter
    base_path: {_BASE_MODEL_PATH}
    adapter_path: {_ADAPTER_PATH}
    prompt_variant: coco_80
    object_field_order: desc_first
    coord_mode: norm1000_text
        """.rstrip()
        if include_adapter
        else ""
    )
    config_path.write_text(
        f"""
run:
  name: raw-text-decode-bias-smoke
  output_dir: {output_dir.as_posix()}
  stages: [{", ".join(selected_stages)}]

study:
  history_scope: full_model_history
  val200_source_jsonl: {source_jsonl.as_posix()}
  val200_source_indices: [0, 1]

decode:
  dataset_variant: lvis_proxy
  pipeline_config: configs/infer/pipeline.yaml
  val200_source_jsonl: {(decode_source_jsonl or source_jsonl).as_posix()}
  device: cuda:0
  semantic_device: cuda:0
  top_p: 0.9
  max_new_tokens: 3084
  batch_size: 4
  seed: 42
  detect_samples: 128
  stop_pressure_mode: {stop_pressure_mode or "suppress_terminating_tokens_after_object_boundary"}
  stop_pressure_min_new_tokens: 0
  stop_pressure_logit_bias: {stop_pressure_logit_bias}
  views: [coco_real, coco_real_strict, coco_real_strict_plausible]
  semantic_model: model_cache/all-MiniLM-L6-v2-local
  semantic_threshold: 0.5
  semantic_batch_size: 64
  num_workers: 8
  metrics: both
  use_segm: false
  strict_parse: false
  lvis_max_dets: 300
  f1ish_iou_thrs: [0.3, 0.5]
  f1ish_pred_scope: annotated

models:
  base_only:
    alias: base_only
    base_path: {_BASE_MODEL_PATH}
    adapter_path: null
    prompt_variant: coco_80
    object_field_order: desc_first
    coord_mode: norm1000_text
{adapter_block}
        """.strip(),
        encoding="utf-8",
    )


def test_load_study_config_parses_raw_text_only_model_block_and_val200_indices(
    tmp_path: Path,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    _write_source_jsonl(source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
    )

    cfg = load_study_config(config_path)

    assert cfg.study.history_scope == "full_model_history"
    assert tuple(cfg.study.val200_source_indices) == (0, 1)
    assert cfg.models.base_only.adapter_path is None
    assert cfg.models.base_only.coord_mode == "norm1000_text"
    assert cfg.models.base_plus_adapter.adapter_path == _ADAPTER_PATH
    assert cfg.models.base_plus_adapter.coord_mode == "norm1000_text"
    assert cfg.decode.dataset_variant == "lvis_proxy"
    assert cfg.decode.pipeline_config == "configs/infer/pipeline.yaml"
    assert cfg.decode.stop_pressure_min_new_tokens == 0
    assert cfg.decode.stop_pressure_logit_bias == 0.0
    assert cfg.decode.views == (
        "coco_real",
        "coco_real_strict",
        "coco_real_strict_plausible",
    )


def test_load_study_config_allows_base_only_only_model_block(tmp_path: Path) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    _write_source_jsonl(source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        include_adapter=False,
    )

    cfg = load_study_config(config_path)

    assert cfg.models.base_only.alias == "base_only"
    assert cfg.models.base_plus_adapter is None


def test_materialize_decode_val200_subset_writes_norm_and_text_pixel_views(
    tmp_path: Path,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=["hydrate", "decode_val200_repeat_penalty"],
    )
    cfg = load_study_config(config_path)

    subset = study_module._materialize_decode_val200_subset(
        cfg=cfg,
        run_dir=tmp_path / "run",
        repo_root=tmp_path,
    )

    subset_dir = tmp_path / "run" / "decode_val200_inputs" / "subset"
    norm_rows = [
        json.loads(line)
        for line in (subset_dir / "sampled.norm.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    text_pixel_rows = [
        json.loads(line)
        for line in (subset_dir / "sampled.text_pixel.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    subset_manifest = json.loads(
        (subset_dir / "subset_manifest.json").read_text(encoding="utf-8")
    )

    assert subset["sampled_norm_jsonl"] == str(subset_dir / "sampled.norm.jsonl")
    assert subset["sampled_text_pixel_jsonl"] == str(
        subset_dir / "sampled.text_pixel.jsonl"
    )
    assert not (subset_dir / "sampled.coord.jsonl").exists()
    assert [row["image_id"] for row in norm_rows] == [101, 202]
    assert text_pixel_rows[0]["objects"][0]["bbox_2d"] == [10, 20, 30, 40]
    assert text_pixel_rows[1]["objects"][0]["bbox_2d"] == [20, 10, 80, 50]
    assert subset_manifest["dataset_variant"] == "lvis_proxy"
    assert subset_manifest["benchmark_scope"] == "val200"
    assert subset_manifest["source_line_indices"] == [0, 1]
    assert subset_manifest["subset_path"] == str(subset_dir / "sampled.norm.jsonl")
    assert subset_manifest["text_pixel_subset_path"] == str(
        subset_dir / "sampled.text_pixel.jsonl"
    )
    assert subset_manifest["coord_tokens_used_for_generation"] is False


def test_build_decode_infer_overrides_stays_raw_text_only_and_can_enable_stop_pressure(
    tmp_path: Path,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=["hydrate", "decode_val200_stop_pressure"],
        stop_pressure_mode="suppress_special_terminating_tokens_after_object_boundary",
    )
    cfg = load_study_config(config_path)
    overrides = study_module._build_decode_infer_overrides(
        cfg=cfg,
        model_cfg=cfg.models.base_plus_adapter,
        subset_artifacts={
            "sampled_text_pixel_jsonl": str(tmp_path / "subset" / "sampled.text_pixel.jsonl"),
            "root_image_dir": str(tmp_path / "images"),
        },
        run_output_dir=tmp_path / "materialized",
        run_name="decode-run",
        repetition_penalty=1.02,
        stop_pressure_active=True,
    )

    assert overrides["run.output_dir"] == str(tmp_path / "materialized")
    assert overrides["run.name"] == "decode-run"
    assert overrides["run.root_image_dir"] == str(tmp_path / "images")
    assert overrides["infer.gt_jsonl"] == str(tmp_path / "subset" / "sampled.text_pixel.jsonl")
    assert overrides["infer.model_checkpoint"] == _ADAPTER_PATH
    assert overrides["infer.mode"] == "text"
    assert overrides["infer.bbox_format"] == "xyxy"
    assert overrides["infer.pred_coord_mode"] == "norm1000"
    assert overrides["infer.backend.type"] == "hf"
    assert overrides["infer.generation.repetition_penalty"] == 1.02
    assert overrides["infer.generation.stop_pressure.mode"] == (
        "suppress_special_terminating_tokens_after_object_boundary"
    )
    assert overrides["infer.generation.stop_pressure.min_new_tokens"] == 0
    assert overrides["infer.generation.stop_pressure.trigger_rule"] == (
        "raw_text_object_boundary"
    )


def test_build_decode_infer_overrides_supports_first_structural_closure_mode(
    tmp_path: Path,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=["hydrate", "decode_val200_stop_pressure"],
        stop_pressure_mode="suppress_first_structural_closure_after_object_boundary",
    )
    cfg = load_study_config(config_path)
    overrides = study_module._build_decode_infer_overrides(
        cfg=cfg,
        model_cfg=cfg.models.base_only,
        subset_artifacts={
            "sampled_text_pixel_jsonl": str(tmp_path / "subset" / "sampled.text_pixel.jsonl"),
            "root_image_dir": str(tmp_path / "images"),
        },
        run_output_dir=tmp_path / "materialized",
        run_name="decode-run",
        repetition_penalty=1.05,
        stop_pressure_active=True,
    )

    assert overrides["infer.mode"] == "text"
    assert overrides["infer.pred_coord_mode"] == "norm1000"
    assert overrides["infer.generation.stop_pressure.mode"] == (
        "suppress_first_structural_closure_after_object_boundary"
    )
    assert overrides["infer.generation.stop_pressure.trigger_rule"] == (
        "raw_text_object_boundary"
    )
    assert overrides["infer.generation.stop_pressure.min_new_tokens"] == 0


def test_build_decode_infer_overrides_supports_array_branch_continuation_steering_mode(
    tmp_path: Path,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=["hydrate", "decode_val200_stop_pressure"],
        stop_pressure_mode="steer_first_array_branch_to_next_object_after_object_boundary",
        stop_pressure_logit_bias=8.5,
    )
    cfg = load_study_config(config_path)
    overrides = study_module._build_decode_infer_overrides(
        cfg=cfg,
        model_cfg=cfg.models.base_only,
        subset_artifacts={
            "sampled_text_pixel_jsonl": str(tmp_path / "subset" / "sampled.text_pixel.jsonl"),
            "root_image_dir": str(tmp_path / "images"),
        },
        run_output_dir=tmp_path / "materialized",
        run_name="decode-run",
        repetition_penalty=1.05,
        stop_pressure_active=True,
    )

    assert overrides["infer.mode"] == "text"
    assert overrides["infer.pred_coord_mode"] == "norm1000"
    assert overrides["infer.generation.stop_pressure.mode"] == (
        "steer_first_array_branch_to_next_object_after_object_boundary"
    )
    assert overrides["infer.generation.stop_pressure.trigger_rule"] == (
        "raw_text_object_boundary"
    )
    assert overrides["infer.generation.stop_pressure.min_new_tokens"] == 0
    assert overrides["infer.generation.stop_pressure.logit_bias"] == 8.5


def test_build_decode_infer_overrides_supports_bbox_tail_closure_steering_mode(
    tmp_path: Path,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=["hydrate", "decode_val200_stop_pressure"],
        stop_pressure_mode="steer_bbox_tail_closure_to_next_object",
        stop_pressure_logit_bias=8.5,
    )
    cfg = load_study_config(config_path)
    overrides = study_module._build_decode_infer_overrides(
        cfg=cfg,
        model_cfg=cfg.models.base_only,
        subset_artifacts={
            "sampled_text_pixel_jsonl": str(tmp_path / "subset" / "sampled.text_pixel.jsonl"),
            "root_image_dir": str(tmp_path / "images"),
        },
        run_output_dir=tmp_path / "materialized",
        run_name="decode-run",
        repetition_penalty=1.05,
        stop_pressure_active=True,
    )

    assert overrides["infer.mode"] == "text"
    assert overrides["infer.pred_coord_mode"] == "norm1000"
    assert overrides["infer.generation.stop_pressure.mode"] == (
        "steer_bbox_tail_closure_to_next_object"
    )
    assert overrides["infer.generation.stop_pressure.trigger_rule"] == (
        "raw_text_object_boundary"
    )
    assert overrides["infer.generation.stop_pressure.min_new_tokens"] == 0
    assert overrides["infer.generation.stop_pressure.logit_bias"] == 8.5


def test_build_decode_infer_overrides_supports_bbox_tail_then_object_open_steering_mode(
    tmp_path: Path,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=["hydrate", "decode_val200_stop_pressure"],
        stop_pressure_mode="steer_bbox_tail_then_object_open",
        stop_pressure_logit_bias=8.5,
    )
    cfg = load_study_config(config_path)
    overrides = study_module._build_decode_infer_overrides(
        cfg=cfg,
        model_cfg=cfg.models.base_only,
        subset_artifacts={
            "sampled_text_pixel_jsonl": str(tmp_path / "subset" / "sampled.text_pixel.jsonl"),
            "root_image_dir": str(tmp_path / "images"),
        },
        run_output_dir=tmp_path / "materialized",
        run_name="decode-run",
        repetition_penalty=1.05,
        stop_pressure_active=True,
    )

    assert overrides["infer.mode"] == "text"
    assert overrides["infer.pred_coord_mode"] == "norm1000"
    assert overrides["infer.generation.stop_pressure.mode"] == (
        "steer_bbox_tail_then_object_open"
    )
    assert overrides["infer.generation.stop_pressure.trigger_rule"] == (
        "raw_text_object_boundary"
    )
    assert overrides["infer.generation.stop_pressure.min_new_tokens"] == 0
    assert overrides["infer.generation.stop_pressure.logit_bias"] == 8.5


def test_build_decode_infer_overrides_supports_bbox_tail_then_object_open_once_steering_mode(
    tmp_path: Path,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=["hydrate", "decode_val200_stop_pressure"],
        stop_pressure_mode="steer_bbox_tail_then_object_open_once",
        stop_pressure_logit_bias=8.5,
    )
    cfg = load_study_config(config_path)
    overrides = study_module._build_decode_infer_overrides(
        cfg=cfg,
        model_cfg=cfg.models.base_only,
        subset_artifacts={
            "sampled_text_pixel_jsonl": str(tmp_path / "subset" / "sampled.text_pixel.jsonl"),
            "root_image_dir": str(tmp_path / "images"),
        },
        run_output_dir=tmp_path / "materialized",
        run_name="decode-run",
        repetition_penalty=1.05,
        stop_pressure_active=True,
    )

    assert overrides["infer.mode"] == "text"
    assert overrides["infer.pred_coord_mode"] == "norm1000"
    assert overrides["infer.generation.stop_pressure.mode"] == (
        "steer_bbox_tail_then_object_open_once"
    )
    assert overrides["infer.generation.stop_pressure.trigger_rule"] == (
        "raw_text_object_boundary"
    )
    assert overrides["infer.generation.stop_pressure.min_new_tokens"] == 0
    assert overrides["infer.generation.stop_pressure.logit_bias"] == 8.5


def test_run_decode_eval_workflow_writes_configs_and_summarizes_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import src.eval.confidence_postop as confidence_postop_module
    import src.eval.proxy_eval_bundle as proxy_eval_bundle_module
    import src.infer.pipeline as infer_pipeline_module

    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    pipeline_config = tmp_path / "configs" / "infer" / "pipeline.yaml"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    pipeline_config.parent.mkdir(parents=True, exist_ok=True)
    pipeline_config.write_text(
        yaml.safe_dump(
            {
                "run": {"name": "placeholder", "output_dir": str(tmp_path / "output")},
                "stages": {"infer": True, "eval": False, "vis": False},
                "infer": {
                    "gt_jsonl": str(decode_source_jsonl),
                    "model_checkpoint": _BASE_MODEL_PATH,
                    "mode": "text",
                    "pred_coord_mode": "norm1000",
                    "backend": {"type": "hf"},
                    "generation": {
                        "temperature": 0.0,
                        "top_p": 0.9,
                        "max_new_tokens": 3084,
                        "repetition_penalty": 1.05,
                        "batch_size": 4,
                        "seed": 42,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=["hydrate", "decode_val200_repeat_penalty"],
        stop_pressure_mode="suppress_special_terminating_tokens_after_object_boundary",
    )
    config_payload = config_path.read_text(encoding="utf-8").replace(
        "configs/infer/pipeline.yaml",
        pipeline_config.as_posix(),
    )
    config_path.write_text(config_payload, encoding="utf-8")
    cfg = load_study_config(config_path)
    subset_artifacts = study_module._materialize_decode_val200_subset(
        cfg=cfg,
        run_dir=tmp_path / "run",
        repo_root=tmp_path,
    )

    def _fake_run_pipeline(*, config_path: Path, overrides):
        assert config_path == pipeline_config
        run_dir = Path(str(overrides["run.output_dir"])) / str(overrides["run.name"])
        run_dir.mkdir(parents=True, exist_ok=True)
        gt_vs_pred_path = run_dir / "gt_vs_pred.jsonl"
        gt_vs_pred_path.write_text(
            json.dumps(
                {
                    "errors": [],
                    "error_entries": [],
                    "pred": [
                        {"desc": "cup", "points": [10, 20, 30, 40], "type": "bbox_2d"}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        pred_token_trace_path = run_dir / "pred_token_trace.jsonl"
        pred_token_trace_path.write_text("{}\n", encoding="utf-8")
        summary_path = run_dir / "summary.json"
        summary_path.write_text("{}\n", encoding="utf-8")
        resolved_config_path = run_dir / "resolved_config.json"
        resolved_config_path.write_text("{}\n", encoding="utf-8")
        return types.SimpleNamespace(
            run_dir=run_dir,
            gt_vs_pred_jsonl=gt_vs_pred_path,
            pred_token_trace_jsonl=pred_token_trace_path,
            summary_json=summary_path,
        )

    def _fake_run_confidence_postop_from_config(cfg_map):
        run_dir = Path(str(cfg_map["artifacts"]["run_dir"]))
        summary_path = run_dir / "confidence_postop_summary.json"
        summary_path.write_text(
            json.dumps({"confidence_method": "bbox_logprob_confidence_exp"}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "gt_vs_pred_scored.jsonl").write_text("{}\n", encoding="utf-8")
        return {"confidence_method": "bbox_logprob_confidence_exp"}

    def _fake_run_proxy_eval_bundle(artifacts, *, options):
        del options
        artifacts.summary_json.write_text(
            json.dumps(
                {
                    "views": {
                        "coco_real": {
                            "metrics": {
                                "bbox_AP": 0.33,
                                "bbox_AP50": 0.44,
                                "bbox_AP75": 0.22,
                                "f1ish@0.50_full_micro": 0.55,
                            }
                        }
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "views": {
                "coco_real": {
                    "metrics": {
                        "bbox_AP": 0.33,
                        "bbox_AP50": 0.44,
                        "bbox_AP75": 0.22,
                        "f1ish@0.50_full_micro": 0.55,
                    }
                }
            }
        }

    monkeypatch.setattr(infer_pipeline_module, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(
        confidence_postop_module,
        "run_confidence_postop_from_config",
        _fake_run_confidence_postop_from_config,
    )
    monkeypatch.setattr(
        proxy_eval_bundle_module,
        "run_proxy_eval_bundle",
        _fake_run_proxy_eval_bundle,
    )

    row = study_module._run_decode_eval_workflow(
        cfg=cfg,
        repo_root=tmp_path,
        stage_dir=tmp_path / "run" / "decode_val200_repeat_penalty",
        stage_name="decode_val200_repeat_penalty",
        model_cfg=cfg.models.base_plus_adapter,
        subset_artifacts=subset_artifacts,
        run_name="rp_1_05",
        repetition_penalty=1.05,
        stop_pressure_active=True,
    )

    assert row["dataset_variant"] == "lvis_proxy"
    assert row["model_alias"] == "base_plus_adapter"
    assert row["repetition_penalty"] == 1.05
    assert row["stop_pressure_active"] is True
    assert (
        row["stop_pressure_mode"]
        == "suppress_special_terminating_tokens_after_object_boundary"
    )
    assert row["coco_real_bbox_AP"] == 0.33
    assert row["parse_valid_rate"] == 1.0
    infer_run_spec = yaml.safe_load(Path(str(row["infer_run_spec_path"])).read_text())
    assert infer_run_spec["overrides"]["infer.mode"] == "text"
    assert (
        infer_run_spec["overrides"]["infer.generation.stop_pressure.mode"]
        == "suppress_special_terminating_tokens_after_object_boundary"
    )
    assert Path(str(row["confidence_config_path"])).is_file()
    assert Path(str(row["proxy_eval_config_path"])).is_file()


def test_run_decode_eval_workflow_records_first_structural_closure_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import src.eval.confidence_postop as confidence_postop_module
    import src.eval.proxy_eval_bundle as proxy_eval_bundle_module
    import src.infer.pipeline as infer_pipeline_module

    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    pipeline_config = tmp_path / "configs" / "infer" / "pipeline.yaml"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    pipeline_config.parent.mkdir(parents=True, exist_ok=True)
    pipeline_config.write_text(
        yaml.safe_dump(
            {
                "run": {"name": "placeholder", "output_dir": str(tmp_path / "output")},
                "stages": {"infer": True, "eval": False, "vis": False},
                "infer": {
                    "gt_jsonl": str(decode_source_jsonl),
                    "model_checkpoint": _BASE_MODEL_PATH,
                    "mode": "text",
                    "pred_coord_mode": "norm1000",
                    "backend": {"type": "hf"},
                    "generation": {
                        "temperature": 0.0,
                        "top_p": 0.9,
                        "max_new_tokens": 3084,
                        "repetition_penalty": 1.05,
                        "batch_size": 4,
                        "seed": 42,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=["hydrate", "decode_val200_repeat_penalty"],
        stop_pressure_mode="suppress_first_structural_closure_after_object_boundary",
    )
    config_payload = config_path.read_text(encoding="utf-8").replace(
        "configs/infer/pipeline.yaml",
        pipeline_config.as_posix(),
    )
    config_path.write_text(config_payload, encoding="utf-8")
    cfg = load_study_config(config_path)
    subset_artifacts = study_module._materialize_decode_val200_subset(
        cfg=cfg,
        run_dir=tmp_path / "run",
        repo_root=tmp_path,
    )

    def _fake_run_pipeline(*, config_path: Path, overrides):
        assert config_path == pipeline_config
        run_dir = Path(str(overrides["run.output_dir"])) / str(overrides["run.name"])
        run_dir.mkdir(parents=True, exist_ok=True)
        gt_vs_pred_path = run_dir / "gt_vs_pred.jsonl"
        gt_vs_pred_path.write_text(
            json.dumps(
                {
                    "errors": [],
                    "error_entries": [],
                    "pred": [
                        {"desc": "cup", "points": [10, 20, 30, 40], "type": "bbox_2d"}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        pred_token_trace_path = run_dir / "pred_token_trace.jsonl"
        pred_token_trace_path.write_text("{}\n", encoding="utf-8")
        summary_path = run_dir / "summary.json"
        summary_path.write_text("{}\n", encoding="utf-8")
        resolved_config_path = run_dir / "resolved_config.json"
        resolved_config_path.write_text("{}\n", encoding="utf-8")
        return types.SimpleNamespace(
            run_dir=run_dir,
            gt_vs_pred_jsonl=gt_vs_pred_path,
            pred_token_trace_jsonl=pred_token_trace_path,
            summary_json=summary_path,
        )

    def _fake_run_confidence_postop_from_config(cfg_map):
        run_dir = Path(str(cfg_map["artifacts"]["run_dir"]))
        summary_path = run_dir / "confidence_postop_summary.json"
        summary_path.write_text(
            json.dumps({"confidence_method": "bbox_logprob_confidence_exp"}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "gt_vs_pred_scored.jsonl").write_text("{}\n", encoding="utf-8")
        return {"confidence_method": "bbox_logprob_confidence_exp"}

    def _fake_run_proxy_eval_bundle(artifacts, *, options):
        del options
        artifacts.summary_json.write_text(
            json.dumps(
                {
                    "views": {
                        "coco_real": {
                            "metrics": {
                                "bbox_AP": 0.21,
                                "bbox_AP50": 0.31,
                                "bbox_AP75": 0.11,
                                "f1ish@0.50_full_micro": 0.41,
                            }
                        }
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "views": {
                "coco_real": {
                    "metrics": {
                        "bbox_AP": 0.21,
                        "bbox_AP50": 0.31,
                        "bbox_AP75": 0.11,
                        "f1ish@0.50_full_micro": 0.41,
                    }
                }
            }
        }

    monkeypatch.setattr(infer_pipeline_module, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(
        confidence_postop_module,
        "run_confidence_postop_from_config",
        _fake_run_confidence_postop_from_config,
    )
    monkeypatch.setattr(
        proxy_eval_bundle_module,
        "run_proxy_eval_bundle",
        _fake_run_proxy_eval_bundle,
    )

    row = study_module._run_decode_eval_workflow(
        cfg=cfg,
        repo_root=tmp_path,
        stage_dir=tmp_path / "run" / "decode_val200_repeat_penalty",
        stage_name="decode_val200_repeat_penalty",
        model_cfg=cfg.models.base_only,
        subset_artifacts=subset_artifacts,
        run_name="rp_1_05",
        repetition_penalty=1.05,
        stop_pressure_active=True,
    )

    assert row["stop_pressure_active"] is True
    assert (
        row["stop_pressure_mode"]
        == "suppress_first_structural_closure_after_object_boundary"
    )
    infer_run_spec = yaml.safe_load(Path(str(row["infer_run_spec_path"])).read_text())
    assert (
        infer_run_spec["overrides"]["infer.generation.stop_pressure.mode"]
        == "suppress_first_structural_closure_after_object_boundary"
    )


def test_run_decode_eval_workflow_records_array_branch_continuation_steering_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import src.eval.confidence_postop as confidence_postop_module
    import src.eval.proxy_eval_bundle as proxy_eval_bundle_module
    import src.infer.pipeline as infer_pipeline_module

    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    pipeline_config = tmp_path / "configs" / "infer" / "pipeline.yaml"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    pipeline_config.parent.mkdir(parents=True, exist_ok=True)
    pipeline_config.write_text(
        yaml.safe_dump(
            {
                "run": {"name": "placeholder", "output_dir": str(tmp_path / "output")},
                "stages": {"infer": True, "eval": False, "vis": False},
                "infer": {
                    "gt_jsonl": str(decode_source_jsonl),
                    "model_checkpoint": _BASE_MODEL_PATH,
                    "mode": "text",
                    "pred_coord_mode": "norm1000",
                    "backend": {"type": "hf"},
                    "generation": {
                        "temperature": 0.0,
                        "top_p": 0.9,
                        "max_new_tokens": 3084,
                        "repetition_penalty": 1.05,
                        "batch_size": 4,
                        "seed": 42,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=["hydrate", "decode_val200_repeat_penalty"],
        stop_pressure_mode="steer_first_array_branch_to_next_object_after_object_boundary",
        stop_pressure_logit_bias=8.5,
    )
    config_payload = config_path.read_text(encoding="utf-8").replace(
        "configs/infer/pipeline.yaml",
        pipeline_config.as_posix(),
    )
    config_path.write_text(config_payload, encoding="utf-8")
    cfg = load_study_config(config_path)
    subset_artifacts = study_module._materialize_decode_val200_subset(
        cfg=cfg,
        run_dir=tmp_path / "run",
        repo_root=tmp_path,
    )

    def _fake_run_pipeline(*, config_path: Path, overrides):
        assert config_path == pipeline_config
        run_dir = Path(str(overrides["run.output_dir"])) / str(overrides["run.name"])
        run_dir.mkdir(parents=True, exist_ok=True)
        gt_vs_pred_path = run_dir / "gt_vs_pred.jsonl"
        gt_vs_pred_path.write_text(
            json.dumps(
                {
                    "errors": [],
                    "error_entries": [],
                    "pred": [
                        {"desc": "cup", "points": [10, 20, 30, 40], "type": "bbox_2d"}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        pred_token_trace_path = run_dir / "pred_token_trace.jsonl"
        pred_token_trace_path.write_text("{}\n", encoding="utf-8")
        summary_path = run_dir / "summary.json"
        summary_path.write_text("{}\n", encoding="utf-8")
        resolved_config_path = run_dir / "resolved_config.json"
        resolved_config_path.write_text("{}\n", encoding="utf-8")
        return types.SimpleNamespace(
            run_dir=run_dir,
            gt_vs_pred_jsonl=gt_vs_pred_path,
            pred_token_trace_jsonl=pred_token_trace_path,
            summary_json=summary_path,
        )

    def _fake_run_confidence_postop_from_config(cfg_map):
        run_dir = Path(str(cfg_map["artifacts"]["run_dir"]))
        summary_path = run_dir / "confidence_postop_summary.json"
        summary_path.write_text(
            json.dumps({"confidence_method": "bbox_logprob_confidence_exp"}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "gt_vs_pred_scored.jsonl").write_text("{}\n", encoding="utf-8")
        return {"confidence_method": "bbox_logprob_confidence_exp"}

    def _fake_run_proxy_eval_bundle(artifacts, *, options):
        del options
        artifacts.summary_json.write_text(
            json.dumps(
                {
                    "views": {
                        "coco_real": {
                            "metrics": {
                                "bbox_AP": 0.24,
                                "bbox_AP50": 0.34,
                                "bbox_AP75": 0.14,
                                "f1ish@0.50_full_micro": 0.44,
                            }
                        }
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "views": {
                "coco_real": {
                    "metrics": {
                        "bbox_AP": 0.24,
                        "bbox_AP50": 0.34,
                        "bbox_AP75": 0.14,
                        "f1ish@0.50_full_micro": 0.44,
                    }
                }
            }
        }

    monkeypatch.setattr(infer_pipeline_module, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(
        confidence_postop_module,
        "run_confidence_postop_from_config",
        _fake_run_confidence_postop_from_config,
    )
    monkeypatch.setattr(
        proxy_eval_bundle_module,
        "run_proxy_eval_bundle",
        _fake_run_proxy_eval_bundle,
    )

    row = study_module._run_decode_eval_workflow(
        cfg=cfg,
        repo_root=tmp_path,
        stage_dir=tmp_path / "run" / "decode_val200_repeat_penalty",
        stage_name="decode_val200_repeat_penalty",
        model_cfg=cfg.models.base_only,
        subset_artifacts=subset_artifacts,
        run_name="rp_1_05",
        repetition_penalty=1.05,
        stop_pressure_active=True,
    )

    assert row["stop_pressure_active"] is True
    assert (
        row["stop_pressure_mode"]
        == "steer_first_array_branch_to_next_object_after_object_boundary"
    )
    assert row["stop_pressure_logit_bias"] == 8.5
    infer_run_spec = yaml.safe_load(Path(str(row["infer_run_spec_path"])).read_text())
    assert (
        infer_run_spec["overrides"]["infer.generation.stop_pressure.mode"]
        == "steer_first_array_branch_to_next_object_after_object_boundary"
    )
    assert (
        infer_run_spec["overrides"]["infer.generation.stop_pressure.logit_bias"] == 8.5
    )


def test_hydrate_case_rows_writes_frozen_candidate_texts() -> None:
    hydrated = hydrate_case_rows(
        case_rows=[
            {
                "case_uid": "base_only:val200:0",
                "model_alias": "base_only",
                "source_jsonl": "/tmp/source.jsonl",
                "source_index": 0,
                "image_id": 101,
                "object_field_order": "desc_first",
                "source_row": {
                    "image_id": 101,
                    "width": 100,
                    "height": 100,
                    "objects": [
                        {"desc": "cup", "bbox_2d": [10, 20, 30, 40]},
                        {"desc": "plate", "bbox_2d": [50, 60, 70, 80]},
                    ],
                },
            }
        ]
    )

    assert hydrated[0]["case_uid"] == "base_only:val200:0"
    assert hydrated[0]["hydration_version"] == "raw_text_decode_bias_v1"
    assert hydrated[0]["baseline_assistant_text"].startswith('{"objects": [')
    assert hydrated[0]["stop_now_candidate_text"].startswith(
        hydrated[0]["baseline_assistant_text"]
    )
    assert hydrated[0]["stop_now_candidate_text"].endswith("]}")
    assert hydrated[0]["continue_with_gt_candidate_text"].startswith(
        hydrated[0]["baseline_assistant_text"]
    )
    assert '"desc": "plate"' in hydrated[0]["continue_with_gt_candidate_text"]


def test_run_study_materializes_run_dir_and_hydrated_inputs(tmp_path: Path) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    _write_source_jsonl(source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        stages=["hydrate"],
    )

    result = run_study(config_path)
    run_dir = tmp_path / "raw-text-decode-bias-smoke"

    assert result["run_dir"] == str(run_dir)
    assert (run_dir / "stage_manifest.json").exists()
    assert (run_dir / "counterfactual_inputs" / "hydrated_cases.jsonl").exists()


def test_run_study_materializes_requested_counterfactual_and_report_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    _write_source_jsonl(source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        stages=[
            "hydrate",
            "counterfactual_eos",
            "counterfactual_repeat_penalty",
            "report",
        ],
    )

    def _fake_eos_stage(*, run_dir: Path, hydrated_rows, **_kwargs) -> dict[str, object]:
        eos_dir = run_dir / "counterfactual_eos"
        eos_dir.mkdir(parents=True, exist_ok=True)
        case_rows = [
            {
                "case_uid": hydrated_rows[0]["case_uid"],
                "model_alias": hydrated_rows[0]["model_alias"],
                "eos_branch_logprob": -0.25,
                "first_continue_branch_logprob": -0.50,
                "eos_now_sum_logprob": -0.30,
                "continue_with_gt_sum_logprob": -0.45,
                "continue_minus_eos_sum_logprob": -0.15,
                "stop_now_candidate_text": hydrated_rows[0]["stop_now_candidate_text"],
                "continue_with_gt_candidate_text": hydrated_rows[0][
                    "continue_with_gt_candidate_text"
                ],
            }
        ]
        (eos_dir / "case_rows.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in case_rows),
            encoding="utf-8",
        )
        return {
            "case_row_count": len(case_rows),
            "case_rows_path": str(eos_dir / "case_rows.jsonl"),
        }

    def _fake_repeat_stage(
        *,
        run_dir: Path,
        hydrated_rows,
        **_kwargs,
    ) -> dict[str, object]:
        repeat_dir = run_dir / "counterfactual_repeat_penalty"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        sweep_rows = []
        for penalty in (1.00, 1.02, 1.05, 1.10):
            sweep_rows.append(
                {
                    "case_uid": hydrated_rows[0]["case_uid"],
                    "model_alias": hydrated_rows[0]["model_alias"],
                    "candidate_kind": "continue_with_gt",
                    "repetition_penalty": penalty,
                    "token_group_deltas": {
                        "desc": -0.1,
                        "digit": -0.2,
                        "structure": -0.05,
                    },
                }
            )
        (repeat_dir / "sweep_rows.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in sweep_rows),
            encoding="utf-8",
        )
        return {
            "row_count": len(sweep_rows),
            "sweep_rows_path": str(repeat_dir / "sweep_rows.jsonl"),
        }

    def _fake_report_stage(*, run_dir: Path, **_kwargs) -> dict[str, object]:
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "benchmark_scope": "val200",
            "coord_mode": "norm1000_text",
            "lanes": ["counterfactual_eos", "counterfactual_repeat_penalty"],
        }
        (report_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"summary_path": str(report_dir / "summary.json")}

    monkeypatch.setattr(
        study_module,
        "_materialize_counterfactual_eos_stage",
        _fake_eos_stage,
        raising=False,
    )
    monkeypatch.setattr(
        study_module,
        "_materialize_counterfactual_repeat_penalty_stage",
        _fake_repeat_stage,
        raising=False,
    )
    monkeypatch.setattr(
        study_module,
        "_materialize_report_stage",
        _fake_report_stage,
        raising=False,
    )

    result = run_study(config_path)
    run_dir = Path(str(result["run_dir"]))
    eos_rows_path = run_dir / "counterfactual_eos" / "case_rows.jsonl"
    repeat_rows_path = run_dir / "counterfactual_repeat_penalty" / "sweep_rows.jsonl"
    report_summary_path = run_dir / "report" / "summary.json"
    resolved_config_path = run_dir / "resolved_config.json"
    manifest = json.loads((run_dir / "stage_manifest.json").read_text(encoding="utf-8"))

    assert eos_rows_path.exists()
    eos_rows = [
        json.loads(line)
        for line in eos_rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert eos_rows[0]["eos_branch_logprob"] == -0.25
    assert "continue_with_gt_candidate_text" in eos_rows[0]
    assert "stop_now_candidate_text" in eos_rows[0]

    assert repeat_rows_path.exists()
    repeat_rows = [
        json.loads(line)
        for line in repeat_rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["repetition_penalty"] for row in repeat_rows] == [1.0, 1.02, 1.05, 1.1]
    assert set(repeat_rows[0]["token_group_deltas"]) == {"desc", "digit", "structure"}

    assert report_summary_path.exists()
    report_summary = json.loads(report_summary_path.read_text(encoding="utf-8"))
    assert report_summary["benchmark_scope"] == "val200"
    assert report_summary["coord_mode"] == "norm1000_text"
    assert resolved_config_path.exists()
    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    assert resolved_config["run"]["stages"] == [
        "hydrate",
        "counterfactual_eos",
        "counterfactual_repeat_penalty",
        "report",
    ]
    assert resolved_config["study"]["history_scope"] == "full_model_history"
    assert resolved_config["models"]["base_only"]["coord_mode"] == "norm1000_text"

    assert manifest["materialized_stages"] == [
        "hydrate",
        "counterfactual_eos",
        "counterfactual_repeat_penalty",
        "report",
    ]


def test_run_study_materializes_requested_branchpoint_census_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    _write_source_jsonl(source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        stages=[
            "hydrate",
            "counterfactual_branchpoint_census",
            "report",
        ],
        include_adapter=False,
    )

    def _fake_branchpoint_stage(*, run_dir: Path, hydrated_rows, **_kwargs) -> dict[str, object]:
        branch_dir = run_dir / "counterfactual_branchpoint_census"
        branch_dir.mkdir(parents=True, exist_ok=True)
        case_rows = [
            {
                "case_uid": hydrated_rows[0]["case_uid"],
                "model_alias": hydrated_rows[0]["model_alias"],
                "array_close_branch": {
                    "stop_minus_continue_raw_logprob": 0.25,
                    "group_summaries": {
                        "close_now": {"raw_prob_mass": 0.6},
                        "next_object": {"raw_prob_mass": 0.3},
                    },
                },
                "final_close_branch": {
                    "status": "ok",
                    "group_summaries": {
                        "close_now": {"raw_prob_mass": 0.7},
                        "wrong_schema": {"raw_prob_mass": 0.2},
                    },
                },
            }
        ]
        summary_rows = [
            {
                "model_alias": hydrated_rows[0]["model_alias"],
                "num_cases": 1,
                "array_branch_close_prefers_stop_count": 1,
                "final_close_available_count": 1,
            }
        ]
        (branch_dir / "case_rows.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in case_rows),
            encoding="utf-8",
        )
        (branch_dir / "summary_rows.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in summary_rows),
            encoding="utf-8",
        )
        (branch_dir / "summary.json").write_text(
            json.dumps({"branchpoint_top_k": 10}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "case_row_count": len(case_rows),
            "summary_row_count": len(summary_rows),
            "case_rows_path": str(branch_dir / "case_rows.jsonl"),
            "summary_rows_path": str(branch_dir / "summary_rows.jsonl"),
            "summary_path": str(branch_dir / "summary.json"),
        }

    def _fake_report_stage(*, run_dir: Path, **_kwargs) -> dict[str, object]:
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "benchmark_scope": "val200",
            "coord_mode": "norm1000_text",
            "lanes": ["counterfactual_branchpoint_census"],
        }
        (report_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"summary_path": str(report_dir / "summary.json")}

    monkeypatch.setattr(
        study_module,
        "_materialize_counterfactual_branchpoint_census_stage",
        _fake_branchpoint_stage,
        raising=False,
    )
    monkeypatch.setattr(
        study_module,
        "_materialize_report_stage",
        _fake_report_stage,
        raising=False,
    )

    result = run_study(config_path)
    run_dir = Path(str(result["run_dir"]))
    branchpoint_rows_path = (
        run_dir / "counterfactual_branchpoint_census" / "case_rows.jsonl"
    )
    report_summary_path = run_dir / "report" / "summary.json"
    resolved_config_path = run_dir / "resolved_config.json"
    manifest = json.loads((run_dir / "stage_manifest.json").read_text(encoding="utf-8"))

    assert branchpoint_rows_path.exists()
    branchpoint_rows = [
        json.loads(line)
        for line in branchpoint_rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert branchpoint_rows[0]["array_close_branch"]["stop_minus_continue_raw_logprob"] == 0.25
    assert branchpoint_rows[0]["final_close_branch"]["status"] == "ok"

    assert report_summary_path.exists()
    report_summary = json.loads(report_summary_path.read_text(encoding="utf-8"))
    assert report_summary["lanes"] == ["counterfactual_branchpoint_census"]

    assert resolved_config_path.exists()
    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    assert resolved_config["run"]["stages"] == [
        "hydrate",
        "counterfactual_branchpoint_census",
        "report",
    ]

    assert manifest["materialized_stages"] == [
        "hydrate",
        "counterfactual_branchpoint_census",
        "report",
    ]


def test_build_branchpoint_group_token_ids_routes_close_comma_tokens_to_wrong_schema() -> None:
    class _Tokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {
                "]": 1,
                "],": 2,
                ", {": 3,
                "}": 4,
            }

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = False,
        ) -> str:
            del skip_special_tokens, clean_up_tokenization_spaces
            mapping = {
                1: "]",
                2: "],",
                3: ", {",
                4: "}",
            }
            return "".join(mapping[int(token_id)] for token_id in token_ids)

    grouped = study_module._build_branchpoint_group_token_ids(
        tokenizer=_Tokenizer(),
        branch_kind="array_close",
    )

    assert grouped["close_now"] == [1]
    assert grouped["wrong_schema"] == [2]
    assert grouped["next_object"] == [3]


def test_run_study_materializes_requested_decode_and_report_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=[
            "hydrate",
            "decode_val200_repeat_penalty",
            "decode_val200_stop_pressure",
            "report",
        ],
    )

    def _fake_decode_subset(*, run_dir: Path, **_kwargs) -> dict[str, object]:
        subset_dir = run_dir / "decode_val200_inputs" / "subset"
        subset_dir.mkdir(parents=True, exist_ok=True)
        norm_path = subset_dir / "sampled.norm.jsonl"
        text_pixel_path = subset_dir / "sampled.text_pixel.jsonl"
        manifest_path = subset_dir / "subset_manifest.json"
        norm_path.write_text("{}\n", encoding="utf-8")
        text_pixel_path.write_text("{}\n", encoding="utf-8")
        manifest_path.write_text("{}\n", encoding="utf-8")
        return {
            "sampled_norm_jsonl": str(norm_path),
            "sampled_text_pixel_jsonl": str(text_pixel_path),
            "subset_manifest_path": str(manifest_path),
            "root_image_dir": str(tmp_path / "images"),
            "row_count": 2,
            "source_line_indices": [0, 1],
            "dataset_variant": "lvis_proxy",
        }

    def _fake_decode_repeat_stage(*, run_dir: Path, **_kwargs) -> dict[str, object]:
        stage_dir = run_dir / "decode_val200_repeat_penalty"
        stage_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "model_alias": "base_only",
                "repetition_penalty": 1.0,
                "dataset_variant": "lvis_proxy",
                "coco_real_bbox_AP": 0.31,
            },
            {
                "model_alias": "base_only",
                "repetition_penalty": 1.05,
                "dataset_variant": "lvis_proxy",
                "coco_real_bbox_AP": 0.29,
            },
        ]
        (stage_dir / "summary_rows.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return {
            "row_count": len(rows),
            "summary_rows_path": str(stage_dir / "summary_rows.jsonl"),
        }

    def _fake_decode_stop_stage(*, run_dir: Path, **_kwargs) -> dict[str, object]:
        stage_dir = run_dir / "decode_val200_stop_pressure"
        stage_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "model_alias": "base_plus_adapter",
                "stop_pressure_active": False,
                "dataset_variant": "lvis_proxy",
                "coco_real_bbox_AP": 0.28,
            },
            {
                "model_alias": "base_plus_adapter",
                "stop_pressure_active": True,
                "dataset_variant": "lvis_proxy",
                "coco_real_bbox_AP": 0.30,
            },
        ]
        (stage_dir / "summary_rows.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return {
            "row_count": len(rows),
            "summary_rows_path": str(stage_dir / "summary_rows.jsonl"),
        }

    monkeypatch.setattr(
        study_module,
        "_materialize_decode_val200_subset",
        _fake_decode_subset,
        raising=False,
    )
    monkeypatch.setattr(
        study_module,
        "_materialize_decode_val200_repeat_penalty_stage",
        _fake_decode_repeat_stage,
        raising=False,
    )
    monkeypatch.setattr(
        study_module,
        "_materialize_decode_val200_stop_pressure_stage",
        _fake_decode_stop_stage,
        raising=False,
    )

    result = run_study(config_path)
    run_dir = Path(str(result["run_dir"]))
    manifest = json.loads((run_dir / "stage_manifest.json").read_text(encoding="utf-8"))
    report_summary = json.loads((run_dir / "report" / "summary.json").read_text(encoding="utf-8"))

    assert manifest["materialized_stages"] == [
        "hydrate",
        "decode_val200_repeat_penalty",
        "decode_val200_stop_pressure",
        "report",
    ]
    assert manifest["decode"]["dataset_variant"] == "lvis_proxy"
    assert manifest["decode"]["subset_manifest_path"].endswith("subset_manifest.json")
    assert result["decode_val200_repeat_penalty_result"]["row_count"] == 2
    assert result["decode_val200_stop_pressure_result"]["row_count"] == 2
    assert "decode_val200_repeat_penalty" in report_summary["lanes"]
    assert "decode_val200_stop_pressure" in report_summary["lanes"]


def test_run_study_materializes_decode_and_report_stages_for_base_only_only_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    decode_source_jsonl = tmp_path / "val200_lvis_proxy.norm.jsonl"
    _write_source_jsonl(source_jsonl)
    _write_decode_source_jsonl(decode_source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        decode_source_jsonl=decode_source_jsonl,
        stages=[
            "hydrate",
            "decode_val200_stop_pressure",
            "report",
        ],
        include_adapter=False,
    )

    def _fake_decode_subset(*, run_dir: Path, **_kwargs) -> dict[str, object]:
        subset_dir = run_dir / "decode_val200_inputs" / "subset"
        subset_dir.mkdir(parents=True, exist_ok=True)
        norm_path = subset_dir / "sampled.norm.jsonl"
        text_pixel_path = subset_dir / "sampled.text_pixel.jsonl"
        manifest_path = subset_dir / "subset_manifest.json"
        norm_path.write_text("{}\n", encoding="utf-8")
        text_pixel_path.write_text("{}\n", encoding="utf-8")
        manifest_path.write_text("{}\n", encoding="utf-8")
        return {
            "sampled_norm_jsonl": str(norm_path),
            "sampled_text_pixel_jsonl": str(text_pixel_path),
            "subset_manifest_path": str(manifest_path),
            "root_image_dir": str(tmp_path / "images"),
            "row_count": 2,
            "source_line_indices": [0, 1],
            "dataset_variant": "lvis_proxy",
        }

    def _fake_decode_stop_stage(*, run_dir: Path, **_kwargs) -> dict[str, object]:
        stage_dir = run_dir / "decode_val200_stop_pressure"
        stage_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "model_alias": "base_only",
                "stop_pressure_active": False,
                "dataset_variant": "lvis_proxy",
                "coco_real_bbox_AP": 0.11,
            },
            {
                "model_alias": "base_only",
                "stop_pressure_active": True,
                "dataset_variant": "lvis_proxy",
                "coco_real_bbox_AP": 0.09,
            },
        ]
        (stage_dir / "summary_rows.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return {
            "row_count": len(rows),
            "summary_rows_path": str(stage_dir / "summary_rows.jsonl"),
        }

    monkeypatch.setattr(
        study_module,
        "_materialize_decode_val200_subset",
        _fake_decode_subset,
        raising=False,
    )
    monkeypatch.setattr(
        study_module,
        "_materialize_decode_val200_stop_pressure_stage",
        _fake_decode_stop_stage,
        raising=False,
    )

    result = run_study(config_path)
    run_dir = Path(str(result["run_dir"]))
    manifest = json.loads((run_dir / "stage_manifest.json").read_text(encoding="utf-8"))
    resolved_config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))

    assert manifest["materialized_stages"] == [
        "hydrate",
        "decode_val200_stop_pressure",
        "report",
    ]
    assert [model["alias"] for model in manifest["models"]] == ["base_only"]
    assert set(resolved_config["models"]) == {"base_only"}
    assert result["decode_val200_stop_pressure_result"]["row_count"] == 2


def test_repo_configs_keep_default_full_and_smoke_lightweight() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_cfg = load_study_config(
        repo_root
        / "configs/analysis/raw_text_coordinate_mechanism/decode_bias_default.yaml"
    )
    smoke_cfg = load_study_config(
        repo_root / "configs/analysis/raw_text_coordinate_mechanism/decode_bias_smoke.yaml"
    )

    assert default_cfg.run.stages == (
        "hydrate",
        "counterfactual_eos",
        "counterfactual_branchpoint_census",
        "counterfactual_repeat_penalty",
        "decode_val200_repeat_penalty",
        "decode_val200_stop_pressure",
        "report",
    )
    assert smoke_cfg.run.stages == ("hydrate",)
    assert smoke_cfg.study.val200_source_indices == default_cfg.study.val200_source_indices
    assert default_cfg.decode.dataset_variant == "lvis_proxy"


def test_counterfactual_stage_helpers_can_share_scorer_cache(tmp_path: Path, monkeypatch) -> None:
    source_jsonl = tmp_path / "val200.jsonl"
    _write_source_jsonl(source_jsonl)
    config_path = tmp_path / "study.yaml"
    _write_config(
        config_path=config_path,
        output_dir=tmp_path,
        source_jsonl=source_jsonl,
        stages=["hydrate", "counterfactual_eos", "counterfactual_repeat_penalty"],
    )
    cfg = load_study_config(config_path)
    source_rows = [json.loads(line) for line in source_jsonl.read_text(encoding="utf-8").splitlines()]
    hydrated_rows = hydrate_case_rows(
        case_rows=[
            {
                "case_uid": "base_only:val200:0",
                "model_alias": "base_only",
                "base_path": _BASE_MODEL_PATH,
                "adapter_path": None,
                "coord_mode": "norm1000_text",
                "prompt_variant": "coco_80",
                "object_field_order": "desc_first",
                "source_jsonl": str(source_jsonl),
                "source_index": 0,
                "image_id": 101,
                "source_row": source_rows[0],
            }
        ]
    )

    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    class _Prepared:
        def __init__(self, assistant_start: int) -> None:
            self.assistant_start = assistant_start

    class _FakeScorer:
        def __init__(self) -> None:
            self.tokenizer = _Tokenizer()

        def prepare_example(
            self,
            *,
            image,
            assistant_text: str,
            desc_positions_rel,
            prompt_variant: str,
            object_field_order: str,
        ):
            del image, assistant_text, desc_positions_rel, prompt_variant, object_field_order
            return _Prepared(assistant_start=0)

        def score_prepared_span_token_rows(
            self,
            *,
            prepared,
            image,
            positions,
            repetition_penalty: float = 1.0,
        ):
            del prepared, image
            return [
                {
                    "position": int(position),
                    "assistant_relative_position": int(position),
                    "token_id": int(position),
                    "raw_logprob": -1.0,
                    "processed_logprob": -1.0 + (float(repetition_penalty) - 1.0),
                }
                for position in positions
            ]

    factory_calls: list[str] = []

    def _fake_make_teacher_forced_scorer(*, model_cfg, scoring_cfg):
        del scoring_cfg
        factory_calls.append(model_cfg.alias)
        return _FakeScorer()

    monkeypatch.setattr(
        study_module,
        "_make_teacher_forced_scorer",
        _fake_make_teacher_forced_scorer,
    )
    monkeypatch.setattr(
        study_module,
        "_load_source_image",
        lambda **_kwargs: __import__("PIL.Image").Image.new("RGB", (8, 8), color="white"),
    )

    scorer_cache: dict[str, object] = {}
    study_module._materialize_counterfactual_eos_stage(
        cfg=cfg,
        run_dir=tmp_path / "run",
        hydrated_rows=hydrated_rows,
        source_jsonl_path=source_jsonl,
        source_rows=source_rows,
        scorer_cache=scorer_cache,
    )
    study_module._materialize_counterfactual_repeat_penalty_stage(
        cfg=cfg,
        run_dir=tmp_path / "run",
        hydrated_rows=hydrated_rows,
        source_jsonl_path=source_jsonl,
        source_rows=source_rows,
        scorer_cache=scorer_cache,
    )

    assert factory_calls == ["base_only"]
