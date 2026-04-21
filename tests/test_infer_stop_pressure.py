import types
from pathlib import Path

import pytest
import torch
from PIL import Image

import src.infer.pipeline as infer_pipeline
from src.infer.artifacts import build_infer_resolved_meta, build_infer_summary_payload
from src.infer.backends import generate_hf_batch
from src.infer.engine import GenerationConfig


def _minimal_infer_stage_cfg(
    tmp_path: Path,
    *,
    backend_type: str,
    stop_pressure: dict[str, object],
) -> dict[str, object]:
    return {
        "infer": {
            "gt_jsonl": str(tmp_path / "gt.jsonl"),
            "model_checkpoint": "dummy-model",
            "mode": "text",
            "pred_coord_mode": "auto",
            "backend": {"type": backend_type},
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 32,
                "repetition_penalty": 1.0,
                "batch_size": 1,
                "stop_pressure": dict(stop_pressure),
            },
        }
    }


def _minimal_artifacts(tmp_path: Path) -> infer_pipeline.ResolvedArtifacts:
    return infer_pipeline.ResolvedArtifacts(
        run_dir=tmp_path / "run",
        gt_vs_pred_jsonl=tmp_path / "gt_vs_pred.jsonl",
        pred_token_trace_jsonl=tmp_path / "pred_token_trace.jsonl",
        gt_vs_pred_scored_jsonl=None,
        summary_json=tmp_path / "summary.json",
        eval_dir=tmp_path / "eval",
        vis_dir=tmp_path / "vis",
    )


def test_generation_config_carries_stop_pressure_fields():
    cfg = GenerationConfig(
        stop_pressure_mode="min_new_tokens_after_object_open",
        stop_pressure_min_new_tokens=24,
        stop_pressure_trigger_rule="raw_text_object_open",
    )

    assert cfg.stop_pressure_mode == "min_new_tokens_after_object_open"
    assert cfg.stop_pressure_min_new_tokens == 24
    assert cfg.stop_pressure_trigger_rule == "raw_text_object_open"


def test_build_infer_summary_payload_records_stop_pressure_block():
    owner = types.SimpleNamespace(
        resolved_mode="text",
        requested_mode="text",
        mode_reason="explicit",
        prompt_variant="default",
        bbox_format="norm1000",
        object_field_order="desc_first",
        object_ordering="sorted",
        prompt_template_hash="prompt-hash",
        attn_implementation_requested="sdpa",
        attn_implementation_selected="sdpa",
        cfg=types.SimpleNamespace(
            model_checkpoint="model",
            adapter_checkpoint=None,
            gt_jsonl="gt.jsonl",
            pred_coord_mode="auto",
            device="cpu",
            limit=8,
            distributed_enabled=False,
        ),
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=128,
            repetition_penalty=1.0,
            batch_size=2,
            seed=7,
            stop_pressure_mode="min_new_tokens_after_object_open",
            stop_pressure_min_new_tokens=24,
            stop_pressure_trigger_rule="raw_text_object_open",
        ),
    )
    counters = types.SimpleNamespace(
        to_summary=lambda: {
            "total_read": 2,
            "total_emitted": 2,
            "counters": {},
            "error_codes": [],
        }
    )

    payload = build_infer_summary_payload(
        owner=owner,
        counters=counters,
        backend="hf",
        determinism="seeded",
        batch_size=owner.gen_cfg.batch_size,
    )

    assert payload["generation"]["stop_pressure"] == {
        "mode": "min_new_tokens_after_object_open",
        "min_new_tokens": 24,
        "trigger_rule": "raw_text_object_open",
        "active": True,
    }


def test_build_infer_summary_payload_marks_stop_pressure_inactive_for_vllm():
    owner = types.SimpleNamespace(
        resolved_mode="text",
        requested_mode="text",
        mode_reason="explicit",
        prompt_variant="default",
        bbox_format="norm1000",
        object_field_order="desc_first",
        object_ordering="sorted",
        prompt_template_hash="prompt-hash",
        attn_implementation_requested=None,
        attn_implementation_selected=None,
        cfg=types.SimpleNamespace(
            model_checkpoint="model",
            adapter_checkpoint=None,
            gt_jsonl="gt.jsonl",
            pred_coord_mode="auto",
            device="cpu",
            limit=8,
            distributed_enabled=False,
        ),
        gen_cfg=GenerationConfig(
            stop_pressure_mode="min_new_tokens_after_object_open",
            stop_pressure_min_new_tokens=24,
            stop_pressure_trigger_rule="raw_text_object_open",
        ),
    )
    counters = types.SimpleNamespace(
        to_summary=lambda: {
            "total_read": 1,
            "total_emitted": 1,
            "counters": {},
            "error_codes": [],
        }
    )

    payload = build_infer_summary_payload(
        owner=owner,
        counters=counters,
        backend="vllm",
        determinism="seeded",
        batch_size=1,
    )

    assert payload["generation"]["stop_pressure"]["active"] is False


def test_build_infer_resolved_meta_records_stop_pressure_block():
    owner = types.SimpleNamespace(
        resolved_mode="text",
        requested_mode="text",
        mode_reason="explicit",
        prompt_variant="default",
        bbox_format="norm1000",
        object_field_order="desc_first",
        object_ordering="sorted",
        prompt_template_hash="prompt-hash",
        cfg=types.SimpleNamespace(
            model_checkpoint="model",
            adapter_checkpoint=None,
            checkpoint_mode="full_model",
            requested_model_checkpoint="model",
            requested_adapter_checkpoint=None,
            resolved_base_model_checkpoint="model",
            resolved_adapter_checkpoint=None,
            gt_jsonl="gt.jsonl",
            pred_coord_mode="auto",
            device="cpu",
            limit=8,
            distributed_enabled=False,
        ),
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=128,
            repetition_penalty=1.0,
            batch_size=2,
            seed=7,
            stop_pressure_mode="min_new_tokens_after_object_open",
            stop_pressure_min_new_tokens=24,
            stop_pressure_trigger_rule="raw_text_object_open",
        ),
    )

    payload = build_infer_resolved_meta(
        owner=owner,
        backend="hf",
        batch_size=owner.gen_cfg.batch_size,
        out_path=Path("gt_vs_pred.jsonl"),
        summary_path=Path("summary.json"),
        trace_path=None,
    )

    assert payload["generation"]["stop_pressure"] == {
        "mode": "min_new_tokens_after_object_open",
        "min_new_tokens": 24,
        "trigger_rule": "raw_text_object_open",
        "active": True,
    }


def test_run_infer_stage_rejects_stop_pressure_for_vllm_backend(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        infer_pipeline,
        "resolve_inference_checkpoint",
        lambda **kwargs: types.SimpleNamespace(
            checkpoint_mode="full_model",
            requested_model_checkpoint=kwargs["model_checkpoint"],
            requested_adapter_checkpoint=kwargs.get("adapter_checkpoint"),
            resolved_base_model_checkpoint=kwargs["model_checkpoint"],
            resolved_adapter_checkpoint=None,
        ),
    )

    cfg = _minimal_infer_stage_cfg(
        tmp_path,
        backend_type="vllm",
        stop_pressure={
            "mode": "min_new_tokens_after_object_open",
            "min_new_tokens": 9,
            "trigger_rule": "raw_text_object_open",
        },
    )

    with pytest.raises(ValueError, match="only supported for infer.backend.type=hf"):
        infer_pipeline._run_infer_stage(
            cfg,
            _minimal_artifacts(tmp_path),
            root_image_dir=None,
        )


def test_run_infer_stage_rejects_unknown_stop_pressure_trigger_rule(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        infer_pipeline,
        "resolve_inference_checkpoint",
        lambda **kwargs: types.SimpleNamespace(
            checkpoint_mode="full_model",
            requested_model_checkpoint=kwargs["model_checkpoint"],
            requested_adapter_checkpoint=kwargs.get("adapter_checkpoint"),
            resolved_base_model_checkpoint=kwargs["model_checkpoint"],
            resolved_adapter_checkpoint=None,
        ),
    )

    cfg = _minimal_infer_stage_cfg(
        tmp_path,
        backend_type="hf",
        stop_pressure={
            "mode": "min_new_tokens_after_object_open",
            "min_new_tokens": 9,
            "trigger_rule": "object_open",
        },
    )

    with pytest.raises(ValueError, match="trigger_rule must be 'raw_text_object_open'"):
        infer_pipeline._run_infer_stage(
            cfg,
            _minimal_artifacts(tmp_path),
            root_image_dir=None,
        )


def test_run_infer_stage_requires_positive_min_new_tokens_for_stop_pressure(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        infer_pipeline,
        "resolve_inference_checkpoint",
        lambda **kwargs: types.SimpleNamespace(
            checkpoint_mode="full_model",
            requested_model_checkpoint=kwargs["model_checkpoint"],
            requested_adapter_checkpoint=kwargs.get("adapter_checkpoint"),
            resolved_base_model_checkpoint=kwargs["model_checkpoint"],
            resolved_adapter_checkpoint=None,
        ),
    )

    cfg = _minimal_infer_stage_cfg(
        tmp_path,
        backend_type="hf",
        stop_pressure={
            "mode": "min_new_tokens_after_object_open",
            "min_new_tokens": 0,
            "trigger_rule": "raw_text_object_open",
        },
    )

    with pytest.raises(ValueError, match="min_new_tokens must be > 0"):
        infer_pipeline._run_infer_stage(
            cfg,
            _minimal_artifacts(tmp_path),
            root_image_dir=None,
        )


def test_generate_hf_batch_sets_min_new_tokens_for_targeted_stop_pressure():
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    captured_kwargs: dict[str, object] = {}

    class _DummyTokenizer:
        def decode(self, _token_ids, **_kwargs) -> str:
            return '{"objects": []}'

        def batch_decode(self, _token_ids, **_kwargs):
            return []

    class _DummyProcessor:
        def __init__(self) -> None:
            self.tokenizer = _DummyTokenizer()

        def apply_chat_template(self, _message, **_kwargs) -> str:
            return "prompt"

        def __call__(self, *, text, images, return_tensors, padding):
            assert text == ["prompt"]
            assert len(images) == 1
            assert return_tensors == "pt"
            assert padding is True
            return {
                "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }

    class _DummyModel:
        def generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return torch.tensor([[11, 12, 13]], dtype=torch.long)

    owner = types.SimpleNamespace(
        model=_DummyModel(),
        processor=_DummyProcessor(),
        logger=types.SimpleNamespace(warning=lambda *args, **kwargs: None),
        cfg=types.SimpleNamespace(device="cpu"),
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=32,
            repetition_penalty=1.0,
            batch_size=1,
            seed=5,
            stop_pressure_mode="min_new_tokens_after_object_open",
            stop_pressure_min_new_tokens=9,
            stop_pressure_trigger_rule="raw_text_object_open",
        ),
        _build_messages=lambda _image: [{"role": "user", "content": "describe"}],
    )

    results = generate_hf_batch(
        owner=owner,
        images=[image],
        result_factory=lambda **kwargs: kwargs,
    )

    assert len(results) == 1
    assert captured_kwargs["min_new_tokens"] == 9
