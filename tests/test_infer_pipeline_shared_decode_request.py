from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.infer.pipeline as infer_pipeline


def _artifacts(tmp_path: Path) -> infer_pipeline.ResolvedArtifacts:
    return infer_pipeline.ResolvedArtifacts(
        run_dir=tmp_path,
        gt_vs_pred_jsonl=tmp_path / "gt_vs_pred.jsonl",
        pred_token_trace_jsonl=tmp_path / "pred_token_trace.jsonl",
        gt_vs_pred_scored_jsonl=tmp_path / "gt_vs_pred_scored.jsonl",
        summary_json=tmp_path / "summary.json",
        eval_dir=tmp_path / "eval",
        vis_dir=tmp_path / "vis",
    )


def _base_cfg(tmp_path: Path) -> dict:
    gt_jsonl = tmp_path / "gt.jsonl"
    gt_jsonl.write_text("", encoding="utf-8")
    return {
        "infer": {
            "gt_jsonl": str(gt_jsonl),
            "model_checkpoint": "model",
            "mode": "coord",
            "pred_coord_mode": "auto",
            "backend": {"type": "hf"},
            "generation": {
                "temperature": 0.01,
                "top_p": 0.95,
                "max_new_tokens": 64,
                "repetition_penalty": 1.05,
                "batch_size": 2,
            },
        }
    }


def _resolved_checkpoint() -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_mode="full_model",
        requested_model_checkpoint="model",
        requested_adapter_checkpoint=None,
        resolved_base_model_checkpoint="model",
        resolved_adapter_checkpoint=None,
    )


def test_infer_stage_uses_shared_decode_request_validation(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["infer"]["generation"]["num_beams"] = 4

    with patch.object(
        infer_pipeline,
        "resolve_inference_checkpoint",
        return_value=_resolved_checkpoint(),
    ), patch.object(
        infer_pipeline,
        "validate_compact_coord_token_adapter_contract",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="decode_mode=beam"):
            infer_pipeline._run_infer_stage(
                cfg,
                _artifacts(tmp_path),
                root_image_dir=None,
            )


def test_infer_stage_propagates_decode_request_to_legacy_bridge(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["detection_template"] = {"id": "compact"}
    cfg["infer"]["generation"].update(
        {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 128,
            "repetition_penalty": 1.2,
            "seed": 13,
            "trace_logprobs": True,
        }
    )
    captures: dict = {}

    def _fake_run_offline_inference(*, inference_kwargs, generation_kwargs, logger=None):
        captures["inference_kwargs"] = dict(inference_kwargs)
        captures["generation_kwargs"] = dict(generation_kwargs)
        captures["logger"] = logger
        captures["infer_called"] = True

    with patch.object(
        infer_pipeline,
        "resolve_inference_checkpoint",
        return_value=_resolved_checkpoint(),
    ), patch.object(
        infer_pipeline,
        "validate_compact_coord_token_adapter_contract",
        return_value=None,
    ), patch.object(
        infer_pipeline,
        "run_offline_inference",
        side_effect=_fake_run_offline_inference,
    ):
        infer_pipeline._run_infer_stage(cfg, _artifacts(tmp_path), root_image_dir=None)

    assert captures["generation_kwargs"]["temperature"] == 0.7
    assert captures["generation_kwargs"]["top_p"] == 0.9
    assert captures["generation_kwargs"]["max_new_tokens"] == 128
    assert captures["generation_kwargs"]["repetition_penalty"] == 1.2
    assert captures["generation_kwargs"]["batch_size"] == 2
    assert captures["generation_kwargs"]["seed"] == 13
    assert captures["inference_kwargs"]["detection_template_id"] == "compact"
    assert captures["inference_kwargs"]["prompt_policy_fingerprint"].startswith(
        "prompt_policy:"
    )
    assert captures["inference_kwargs"]["decode_policy_fingerprint"].startswith("decode:")
    assert captures["inference_kwargs"]["model_identity_fingerprint"].startswith(
        "model:"
    )
    assert captures["infer_called"] is True


def test_infer_stage_allows_vllm_trace_logprobs_through_shared_runtime(
    tmp_path: Path,
) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["infer"]["backend"] = {"type": "vllm"}
    cfg["infer"]["generation"]["trace_logprobs"] = True
    captures: dict = {}

    def _fake_run_offline_inference(*, inference_kwargs, generation_kwargs, logger=None):
        captures["inference_kwargs"] = dict(inference_kwargs)
        captures["generation_kwargs"] = dict(generation_kwargs)

    with patch.object(
        infer_pipeline,
        "resolve_inference_checkpoint",
        return_value=_resolved_checkpoint(),
    ), patch.object(
        infer_pipeline,
        "validate_compact_coord_token_adapter_contract",
        return_value=None,
    ), patch.object(
        infer_pipeline,
        "run_offline_inference",
        side_effect=_fake_run_offline_inference,
    ):
        infer_pipeline._run_infer_stage(
            cfg,
            _artifacts(tmp_path),
            root_image_dir=None,
        )

    assert captures["inference_kwargs"]["backend_type"] == "vllm"
    assert captures["generation_kwargs"]["trace_logprobs"] is True


def test_infer_stage_resolves_auto_mode_once_for_runtime_config(
    tmp_path: Path,
) -> None:
    cfg = _base_cfg(tmp_path)
    gt_jsonl = Path(cfg["infer"]["gt_jsonl"])
    gt_jsonl.write_text(
        '{"width":10,"height":10,"objects":[{"bbox_2d":[0,0,9,9],"desc":"cat"}]}\n',
        encoding="utf-8",
    )
    cfg["infer"]["mode"] = "auto"
    captures: dict = {}

    def _fake_run_offline_inference(*, inference_kwargs, generation_kwargs, logger=None):
        captures["inference_kwargs"] = dict(inference_kwargs)
        captures["generation_kwargs"] = dict(generation_kwargs)
        captures["logger"] = logger

    with patch.object(
        infer_pipeline,
        "resolve_inference_checkpoint",
        return_value=_resolved_checkpoint(),
    ), patch.object(
        infer_pipeline,
        "validate_compact_coord_token_adapter_contract",
        return_value=None,
    ), patch.object(
        infer_pipeline,
        "run_offline_inference",
        side_effect=_fake_run_offline_inference,
    ):
        infer_pipeline._run_infer_stage(cfg, _artifacts(tmp_path), root_image_dir=None)

    assert captures["inference_kwargs"]["mode"] == "text"
    assert captures["inference_kwargs"]["requested_mode"] == "auto"
    assert captures["inference_kwargs"]["mode_resolution_reason"] == (
        "within_image_bounds"
    )


def test_run_pipeline_records_runtime_policy_in_resolved_config(tmp_path: Path) -> None:
    from src.infer.artifacts import load_comparable_artifact

    cfg = _base_cfg(tmp_path)
    cfg.update(
        {
            "run": {"name": "demo", "output_dir": str(tmp_path / "out")},
            "stages": {"infer": True, "eval": False, "vis": False},
        }
    )

    with patch.object(infer_pipeline, "_load_yaml", return_value=cfg), patch.object(
        infer_pipeline,
        "resolve_inference_checkpoint",
        return_value=_resolved_checkpoint(),
    ), patch.object(
        infer_pipeline,
        "validate_compact_coord_token_adapter_contract",
        return_value=None,
    ), patch.object(
        infer_pipeline,
        "_run_infer_stage",
        return_value=None,
    ):
        artifacts = infer_pipeline.run_pipeline(config_path=tmp_path / "config.yaml")

    resolved = __import__("json").loads(
        (artifacts.run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    provenance = resolved["inference_provenance"]
    assert provenance["comparable"] is True
    assert provenance["prompt_policy_fingerprint"].startswith("prompt_policy:")
    assert provenance["decode_policy_fingerprint"].startswith("decode:")
    assert provenance["model_identity_fingerprint"].startswith("model:")
    assert provenance["score_policy"] == "none"
    assert "prompt_policy_fingerprint" not in provenance["missing_provenance_fields"]
    assert "model_identity_fingerprint" not in provenance["missing_provenance_fields"]

    artifacts.gt_vs_pred_jsonl.write_text(
        '{"gt":[],"pred":[],"width":1,"height":1}\n',
        encoding="utf-8",
    )
    loaded = load_comparable_artifact(artifacts.gt_vs_pred_jsonl)

    assert loaded["provenance_carrier"].endswith("resolved_config.json")
    assert loaded["provenance_path"].endswith("resolved_config.json")
    assert loaded["provenance"]["inference_provenance"] == provenance
