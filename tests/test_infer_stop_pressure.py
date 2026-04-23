import types
from pathlib import Path

import pytest
import torch
from PIL import Image

import src.infer.engine as infer_engine_module
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
        stop_pressure_logit_bias=0.0,
    )

    assert cfg.stop_pressure_mode == "min_new_tokens_after_object_open"
    assert cfg.stop_pressure_min_new_tokens == 24
    assert cfg.stop_pressure_trigger_rule == "raw_text_object_open"
    assert cfg.stop_pressure_logit_bias == 0.0


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
            stop_pressure_mode=(
                "steer_first_array_branch_to_next_object_after_object_boundary"
            ),
            stop_pressure_min_new_tokens=0,
            stop_pressure_trigger_rule="raw_text_object_boundary",
            stop_pressure_logit_bias=8.5,
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
        "mode": "steer_first_array_branch_to_next_object_after_object_boundary",
        "min_new_tokens": 0,
        "trigger_rule": "raw_text_object_boundary",
        "logit_bias": 8.5,
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
        "logit_bias": 0.0,
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


def test_run_infer_stage_rejects_unknown_stop_pressure_mode(
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
            "mode": "force_continue",
            "min_new_tokens": 9,
            "trigger_rule": "raw_text_object_open",
        },
    )

    with pytest.raises(
        ValueError,
        match="stop_pressure.mode must be 'min_new_tokens_after_object_open'",
    ):
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


def test_run_infer_stage_accepts_special_only_terminator_suppression_mode(
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
    captured: dict[str, object] = {}

    class _DummyInferenceEngine:
        def __init__(self, inf_cfg, gen_cfg) -> None:
            captured["inf_cfg"] = inf_cfg
            captured["gen_cfg"] = gen_cfg

        def infer(self) -> None:
            captured["infer_called"] = True

    monkeypatch.setattr(infer_engine_module, "InferenceEngine", _DummyInferenceEngine)

    cfg = _minimal_infer_stage_cfg(
        tmp_path,
        backend_type="hf",
        stop_pressure={
            "mode": "suppress_special_terminating_tokens_after_object_boundary",
            "min_new_tokens": 0,
            "trigger_rule": "raw_text_object_boundary",
        },
    )

    infer_pipeline._run_infer_stage(
        cfg,
        _minimal_artifacts(tmp_path),
        root_image_dir=None,
    )

    gen_cfg = captured["gen_cfg"]
    assert captured["infer_called"] is True
    assert isinstance(gen_cfg, GenerationConfig)
    assert (
        gen_cfg.stop_pressure_mode
        == "suppress_special_terminating_tokens_after_object_boundary"
    )
    assert gen_cfg.stop_pressure_trigger_rule == "raw_text_object_boundary"
    assert gen_cfg.stop_pressure_active is True


def test_run_infer_stage_accepts_first_structural_closure_suppression_mode(
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
    captured: dict[str, object] = {}

    class _DummyInferenceEngine:
        def __init__(self, inf_cfg, gen_cfg) -> None:
            captured["inf_cfg"] = inf_cfg
            captured["gen_cfg"] = gen_cfg

        def infer(self) -> None:
            captured["infer_called"] = True

    monkeypatch.setattr(infer_engine_module, "InferenceEngine", _DummyInferenceEngine)

    cfg = _minimal_infer_stage_cfg(
        tmp_path,
        backend_type="hf",
        stop_pressure={
            "mode": "suppress_first_structural_closure_after_object_boundary",
            "min_new_tokens": 0,
            "trigger_rule": "raw_text_object_boundary",
        },
    )

    infer_pipeline._run_infer_stage(
        cfg,
        _minimal_artifacts(tmp_path),
        root_image_dir=None,
    )

    gen_cfg = captured["gen_cfg"]
    assert captured["infer_called"] is True
    assert isinstance(gen_cfg, GenerationConfig)
    assert (
        gen_cfg.stop_pressure_mode
        == "suppress_first_structural_closure_after_object_boundary"
    )
    assert gen_cfg.stop_pressure_trigger_rule == "raw_text_object_boundary"
    assert gen_cfg.stop_pressure_active is True


def test_run_infer_stage_accepts_array_branch_continuation_steering_mode(
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
    captured: dict[str, object] = {}

    class _DummyInferenceEngine:
        def __init__(self, inf_cfg, gen_cfg) -> None:
            captured["inf_cfg"] = inf_cfg
            captured["gen_cfg"] = gen_cfg

        def infer(self) -> None:
            captured["infer_called"] = True

    monkeypatch.setattr(infer_engine_module, "InferenceEngine", _DummyInferenceEngine)

    cfg = _minimal_infer_stage_cfg(
        tmp_path,
        backend_type="hf",
        stop_pressure={
            "mode": "steer_first_array_branch_to_next_object_after_object_boundary",
            "min_new_tokens": 0,
            "trigger_rule": "raw_text_object_boundary",
            "logit_bias": 8.5,
        },
    )

    infer_pipeline._run_infer_stage(
        cfg,
        _minimal_artifacts(tmp_path),
        root_image_dir=None,
    )

    gen_cfg = captured["gen_cfg"]
    assert captured["infer_called"] is True
    assert isinstance(gen_cfg, GenerationConfig)
    assert (
        gen_cfg.stop_pressure_mode
        == "steer_first_array_branch_to_next_object_after_object_boundary"
    )
    assert gen_cfg.stop_pressure_trigger_rule == "raw_text_object_boundary"
    assert gen_cfg.stop_pressure_logit_bias == 8.5
    assert gen_cfg.stop_pressure_active is True


def test_run_infer_stage_accepts_bbox_tail_closure_steering_mode(
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
    captured: dict[str, object] = {}

    class _DummyInferenceEngine:
        def __init__(self, inf_cfg, gen_cfg) -> None:
            captured["inf_cfg"] = inf_cfg
            captured["gen_cfg"] = gen_cfg

        def infer(self) -> None:
            captured["infer_called"] = True

    monkeypatch.setattr(infer_engine_module, "InferenceEngine", _DummyInferenceEngine)

    cfg = _minimal_infer_stage_cfg(
        tmp_path,
        backend_type="hf",
        stop_pressure={
            "mode": "steer_bbox_tail_closure_to_next_object",
            "min_new_tokens": 0,
            "trigger_rule": "raw_text_object_boundary",
            "logit_bias": 8.5,
        },
    )

    infer_pipeline._run_infer_stage(
        cfg,
        _minimal_artifacts(tmp_path),
        root_image_dir=None,
    )

    gen_cfg = captured["gen_cfg"]
    assert captured["infer_called"] is True
    assert isinstance(gen_cfg, GenerationConfig)
    assert gen_cfg.stop_pressure_mode == "steer_bbox_tail_closure_to_next_object"
    assert gen_cfg.stop_pressure_trigger_rule == "raw_text_object_boundary"
    assert gen_cfg.stop_pressure_logit_bias == 8.5
    assert gen_cfg.stop_pressure_active is True


def test_run_infer_stage_accepts_bbox_tail_then_object_open_steering_mode(
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
    captured: dict[str, object] = {}

    class _DummyInferenceEngine:
        def __init__(self, inf_cfg, gen_cfg) -> None:
            captured["inf_cfg"] = inf_cfg
            captured["gen_cfg"] = gen_cfg

        def infer(self) -> None:
            captured["infer_called"] = True

    monkeypatch.setattr(infer_engine_module, "InferenceEngine", _DummyInferenceEngine)

    cfg = _minimal_infer_stage_cfg(
        tmp_path,
        backend_type="hf",
        stop_pressure={
            "mode": "steer_bbox_tail_then_object_open",
            "min_new_tokens": 0,
            "trigger_rule": "raw_text_object_boundary",
            "logit_bias": 8.5,
        },
    )

    infer_pipeline._run_infer_stage(
        cfg,
        _minimal_artifacts(tmp_path),
        root_image_dir=None,
    )

    gen_cfg = captured["gen_cfg"]
    assert captured["infer_called"] is True
    assert isinstance(gen_cfg, GenerationConfig)
    assert gen_cfg.stop_pressure_mode == "steer_bbox_tail_then_object_open"
    assert gen_cfg.stop_pressure_trigger_rule == "raw_text_object_boundary"
    assert gen_cfg.stop_pressure_logit_bias == 8.5
    assert gen_cfg.stop_pressure_active is True


def test_run_infer_stage_accepts_bbox_tail_then_object_open_once_steering_mode(
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
    captured: dict[str, object] = {}

    class _DummyInferenceEngine:
        def __init__(self, inf_cfg, gen_cfg) -> None:
            captured["inf_cfg"] = inf_cfg
            captured["gen_cfg"] = gen_cfg

        def infer(self) -> None:
            captured["infer_called"] = True

    monkeypatch.setattr(infer_engine_module, "InferenceEngine", _DummyInferenceEngine)

    cfg = _minimal_infer_stage_cfg(
        tmp_path,
        backend_type="hf",
        stop_pressure={
            "mode": "steer_bbox_tail_then_object_open_once",
            "min_new_tokens": 0,
            "trigger_rule": "raw_text_object_boundary",
            "logit_bias": 8.5,
        },
    )

    infer_pipeline._run_infer_stage(
        cfg,
        _minimal_artifacts(tmp_path),
        root_image_dir=None,
    )

    gen_cfg = captured["gen_cfg"]
    assert captured["infer_called"] is True
    assert isinstance(gen_cfg, GenerationConfig)
    assert gen_cfg.stop_pressure_mode == "steer_bbox_tail_then_object_open_once"
    assert gen_cfg.stop_pressure_trigger_rule == "raw_text_object_boundary"
    assert gen_cfg.stop_pressure_logit_bias == 8.5
    assert gen_cfg.stop_pressure_active is True


def test_run_infer_stage_rejects_stop_pressure_fields_without_mode(
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
            "min_new_tokens": -3,
        },
    )

    with pytest.raises(ValueError, match="stop_pressure.mode is required"):
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


def test_generate_hf_batch_suppresses_terminating_tokens_at_object_boundary():
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    captured_kwargs: dict[str, object] = {}

    class _DummyTokenizer:
        eos_token_id = 99
        _id_to_text = {
            11: "<prompt_a>",
            12: "<prompt_b>",
            20: ",",
            21: "]",
            22: "]}",
            30: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}',
            99: "<|endoftext|>",
        }

        def decode(self, token_ids, **_kwargs) -> str:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return "".join(self._id_to_text[int(tok)] for tok in token_ids)

        def batch_decode(self, token_ids, **_kwargs):
            return [self.decode(ids, **_kwargs) for ids in token_ids]

        def get_vocab(self):
            return {f"tok_{tok_id}": tok_id for tok_id in self._id_to_text}

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
            logits_processor = kwargs.get("logits_processor")
            assert logits_processor is not None
            scores = torch.zeros((1, 100), dtype=torch.float)
            processed = logits_processor(
                torch.tensor([[11, 12, 30]], dtype=torch.long),
                scores.clone(),
            )
            assert torch.isneginf(processed[0, 21])
            assert torch.isneginf(processed[0, 22])
            assert torch.isneginf(processed[0, 99])
            assert processed[0, 20].item() == 0.0
            return torch.tensor([[11, 12, 30, 20]], dtype=torch.long)

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
            stop_pressure_mode="suppress_terminating_tokens_after_object_boundary",
            stop_pressure_min_new_tokens=0,
            stop_pressure_trigger_rule="raw_text_object_boundary",
        ),
        _build_messages=lambda _image: [{"role": "user", "content": "describe"}],
    )

    results = generate_hf_batch(
        owner=owner,
        images=[image],
        result_factory=lambda **kwargs: kwargs,
    )

    assert len(results) == 1
    assert captured_kwargs["logits_processor"] is not None


def test_generate_hf_batch_special_only_mode_suppresses_only_special_terminators():
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    captured_kwargs: dict[str, object] = {}

    class _DummyTokenizer:
        eos_token_id = 99
        _id_to_text = {
            11: "<prompt_a>",
            12: "<prompt_b>",
            21: "]",
            22: "]}",
            23: "<|im_end|>",
            30: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}',
            99: "<|endoftext|>",
        }

        def decode(self, token_ids, **_kwargs) -> str:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return "".join(self._id_to_text[int(tok)] for tok in token_ids)

        def batch_decode(self, token_ids, **_kwargs):
            return [self.decode(ids, **_kwargs) for ids in token_ids]

        def get_vocab(self):
            return {f"tok_{tok_id}": tok_id for tok_id in self._id_to_text}

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
            logits_processor = kwargs.get("logits_processor")
            assert logits_processor is not None
            scores = torch.zeros((1, 100), dtype=torch.float)
            processed = logits_processor(
                torch.tensor([[11, 12, 30]], dtype=torch.long),
                scores.clone(),
            )
            assert not torch.isneginf(processed[0, 21])
            assert not torch.isneginf(processed[0, 22])
            assert torch.isneginf(processed[0, 23])
            assert torch.isneginf(processed[0, 99])
            return torch.tensor([[11, 12, 30, 21]], dtype=torch.long)

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
            stop_pressure_mode="suppress_special_terminating_tokens_after_object_boundary",
            stop_pressure_min_new_tokens=0,
            stop_pressure_trigger_rule="raw_text_object_boundary",
        ),
        _build_messages=lambda _image: [{"role": "user", "content": "describe"}],
    )

    results = generate_hf_batch(
        owner=owner,
        images=[image],
        result_factory=lambda **kwargs: kwargs,
    )

    assert len(results) == 1
    assert captured_kwargs["logits_processor"] is not None


def test_generate_hf_batch_first_structural_closure_mode_is_local_to_fresh_boundary():
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    captured_kwargs: dict[str, object] = {}

    class _DummyTokenizer:
        eos_token_id = 99
        _id_to_text = {
            11: "<prompt_a>",
            12: "<prompt_b>",
            21: "]",
            22: "]}",
            23: "<|im_end|>",
            30: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}',
            40: '"',
            99: "<|endoftext|>",
        }

        def decode(self, token_ids, **_kwargs) -> str:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return "".join(self._id_to_text[int(tok)] for tok in token_ids)

        def batch_decode(self, token_ids, **_kwargs):
            return [self.decode(ids, **_kwargs) for ids in token_ids]

        def get_vocab(self):
            return {f"tok_{tok_id}": tok_id for tok_id in self._id_to_text}

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
            logits_processor = kwargs.get("logits_processor")
            assert logits_processor is not None
            scores = torch.zeros((1, 100), dtype=torch.float)

            processed_fresh = logits_processor(
                torch.tensor([[11, 12, 30]], dtype=torch.long),
                scores.clone(),
            )
            assert torch.isneginf(processed_fresh[0, 21])
            assert torch.isneginf(processed_fresh[0, 22])
            assert not torch.isneginf(processed_fresh[0, 23])
            assert not torch.isneginf(processed_fresh[0, 99])

            processed_dirty = logits_processor(
                torch.tensor([[11, 12, 30, 40]], dtype=torch.long),
                scores.clone(),
            )
            assert not torch.isneginf(processed_dirty[0, 21])
            assert not torch.isneginf(processed_dirty[0, 22])
            assert not torch.isneginf(processed_dirty[0, 23])
            assert not torch.isneginf(processed_dirty[0, 99])

            return torch.tensor([[11, 12, 30, 40]], dtype=torch.long)

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
            stop_pressure_mode="suppress_first_structural_closure_after_object_boundary",
            stop_pressure_min_new_tokens=0,
            stop_pressure_trigger_rule="raw_text_object_boundary",
        ),
        _build_messages=lambda _image: [{"role": "user", "content": "describe"}],
    )

    results = generate_hf_batch(
        owner=owner,
        images=[image],
        result_factory=lambda **kwargs: kwargs,
    )

    assert len(results) == 1
    assert captured_kwargs["logits_processor"] is not None


def test_generate_hf_batch_array_branch_continuation_steering_is_local_and_positive():
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    captured_kwargs: dict[str, object] = {}

    class _DummyTokenizer:
        eos_token_id = 99
        _id_to_text = {
            11: "<prompt_a>",
            12: "<prompt_b>",
            21: "]",
            22: "]}",
            23: "],",
            24: "]}\"",
            25: ",",
            26: " ,",
            27: "<|im_end|>",
            30: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}',
            40: '"',
            99: "<|endoftext|>",
        }

        def decode(self, token_ids, **_kwargs) -> str:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return "".join(self._id_to_text[int(tok)] for tok in token_ids)

        def batch_decode(self, token_ids, **_kwargs):
            return [self.decode(ids, **_kwargs) for ids in token_ids]

        def get_vocab(self):
            return {f"tok_{tok_id}": tok_id for tok_id in self._id_to_text}

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
            logits_processor = kwargs.get("logits_processor")
            assert logits_processor is not None
            scores = torch.zeros((1, 100), dtype=torch.float)

            processed_fresh = logits_processor(
                torch.tensor([[11, 12, 30]], dtype=torch.long),
                scores.clone(),
            )
            assert torch.isneginf(processed_fresh[0, 21])
            assert torch.isneginf(processed_fresh[0, 22])
            assert torch.isneginf(processed_fresh[0, 23])
            assert torch.isneginf(processed_fresh[0, 24])
            assert processed_fresh[0, 25].item() == pytest.approx(8.5)
            assert processed_fresh[0, 26].item() == pytest.approx(8.5)
            assert not torch.isneginf(processed_fresh[0, 27])
            assert not torch.isneginf(processed_fresh[0, 99])

            processed_dirty = logits_processor(
                torch.tensor([[11, 12, 30, 40]], dtype=torch.long),
                scores.clone(),
            )
            assert not torch.isneginf(processed_dirty[0, 21])
            assert not torch.isneginf(processed_dirty[0, 22])
            assert not torch.isneginf(processed_dirty[0, 23])
            assert not torch.isneginf(processed_dirty[0, 24])
            assert processed_dirty[0, 25].item() == pytest.approx(0.0)
            assert processed_dirty[0, 26].item() == pytest.approx(0.0)

            return torch.tensor([[11, 12, 30, 25]], dtype=torch.long)

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
            stop_pressure_mode=(
                "steer_first_array_branch_to_next_object_after_object_boundary"
            ),
            stop_pressure_min_new_tokens=0,
            stop_pressure_trigger_rule="raw_text_object_boundary",
            stop_pressure_logit_bias=8.5,
        ),
        _build_messages=lambda _image: [{"role": "user", "content": "describe"}],
    )

    results = generate_hf_batch(
        owner=owner,
        images=[image],
        result_factory=lambda **kwargs: kwargs,
    )

    assert len(results) == 1
    assert captured_kwargs["logits_processor"] is not None


def test_generate_hf_batch_bbox_tail_closure_steering_targets_fused_close_tokens():
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    captured_kwargs: dict[str, object] = {}

    class _DummyTokenizer:
        eos_token_id = 99
        _id_to_text = {
            11: "<prompt_a>",
            12: "<prompt_b>",
            21: "]}",
            22: "]},",
            23: "],",
            24: "]}\"",
            25: ",",
            30: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4',
            31: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}',
            99: "<|endoftext|>",
        }

        def decode(self, token_ids, **_kwargs) -> str:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return "".join(self._id_to_text[int(tok)] for tok in token_ids)

        def batch_decode(self, token_ids, **_kwargs):
            return [self.decode(ids, **_kwargs) for ids in token_ids]

        def get_vocab(self):
            return {f"tok_{tok_id}": tok_id for tok_id in self._id_to_text}

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
            logits_processor = kwargs.get("logits_processor")
            assert logits_processor is not None
            scores = torch.zeros((1, 100), dtype=torch.float)

            processed_bbox_tail = logits_processor(
                torch.tensor([[11, 12, 30]], dtype=torch.long),
                scores.clone(),
            )
            assert torch.isneginf(processed_bbox_tail[0, 21])
            assert processed_bbox_tail[0, 22].item() == pytest.approx(8.5)
            assert torch.isneginf(processed_bbox_tail[0, 23])
            assert torch.isneginf(processed_bbox_tail[0, 24])
            assert processed_bbox_tail[0, 25].item() == pytest.approx(0.0)

            processed_after_close = logits_processor(
                torch.tensor([[11, 12, 31]], dtype=torch.long),
                scores.clone(),
            )
            assert not torch.isneginf(processed_after_close[0, 21])
            assert processed_after_close[0, 22].item() == pytest.approx(0.0)
            assert not torch.isneginf(processed_after_close[0, 23])
            assert not torch.isneginf(processed_after_close[0, 24])

            return torch.tensor([[11, 12, 30, 22]], dtype=torch.long)

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
            stop_pressure_mode="steer_bbox_tail_closure_to_next_object",
            stop_pressure_min_new_tokens=0,
            stop_pressure_trigger_rule="raw_text_object_boundary",
            stop_pressure_logit_bias=8.5,
        ),
        _build_messages=lambda _image: [{"role": "user", "content": "describe"}],
    )

    results = generate_hf_batch(
        owner=owner,
        images=[image],
        result_factory=lambda **kwargs: kwargs,
    )

    assert len(results) == 1
    assert captured_kwargs["logits_processor"] is not None


def test_generate_hf_batch_bbox_tail_then_object_open_steering_targets_followup_open_tokens():
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    captured_kwargs: dict[str, object] = {}

    class _DummyTokenizer:
        eos_token_id = 99
        _id_to_text = {
            11: "<prompt_a>",
            12: "<prompt_b>",
            21: "]}",
            22: "]},",
            23: "],",
            24: "]}\"",
            25: ",",
            26: " {\n",
            27: " {\"",
            28: " \"",
            29: "poly",
            30: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4',
            31: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}',
            32: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]},\n     ',
            33: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]},\n     {\n',
            99: "<|endoftext|>",
        }

        def decode(self, token_ids, **_kwargs) -> str:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return "".join(self._id_to_text[int(tok)] for tok in token_ids)

        def batch_decode(self, token_ids, **_kwargs):
            return [self.decode(ids, **_kwargs) for ids in token_ids]

        def get_vocab(self):
            return {f"tok_{tok_id}": tok_id for tok_id in self._id_to_text}

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
            logits_processor = kwargs.get("logits_processor")
            assert logits_processor is not None
            scores = torch.zeros((1, 100), dtype=torch.float)

            processed_bbox_tail = logits_processor(
                torch.tensor([[11, 12, 30]], dtype=torch.long),
                scores.clone(),
            )
            assert torch.isneginf(processed_bbox_tail[0, 21])
            assert processed_bbox_tail[0, 22].item() == pytest.approx(8.5)
            assert torch.isneginf(processed_bbox_tail[0, 23])
            assert torch.isneginf(processed_bbox_tail[0, 24])
            assert processed_bbox_tail[0, 25].item() == pytest.approx(0.0)

            processed_followup = logits_processor(
                torch.tensor([[11, 12, 32]], dtype=torch.long),
                scores.clone(),
            )
            assert processed_followup[0, 26].item() == pytest.approx(8.5)
            assert processed_followup[0, 27].item() == pytest.approx(8.5)
            assert torch.isneginf(processed_followup[0, 28])
            assert processed_followup[0, 29].item() == pytest.approx(0.0)
            assert not torch.isneginf(processed_followup[0, 99])

            processed_after_open = logits_processor(
                torch.tensor([[11, 12, 33]], dtype=torch.long),
                scores.clone(),
            )
            assert processed_after_open[0, 26].item() == pytest.approx(0.0)
            assert processed_after_open[0, 27].item() == pytest.approx(0.0)
            assert not torch.isneginf(processed_after_open[0, 28])

            return torch.tensor([[11, 12, 30, 22]], dtype=torch.long)

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
            stop_pressure_mode="steer_bbox_tail_then_object_open",
            stop_pressure_min_new_tokens=0,
            stop_pressure_trigger_rule="raw_text_object_boundary",
            stop_pressure_logit_bias=8.5,
        ),
        _build_messages=lambda _image: [{"role": "user", "content": "describe"}],
    )

    results = generate_hf_batch(
        owner=owner,
        images=[image],
        result_factory=lambda **kwargs: kwargs,
    )

    assert len(results) == 1
    assert captured_kwargs["logits_processor"] is not None


def test_generate_hf_batch_bbox_tail_then_object_open_once_turns_off_after_next_object_opens():
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))
    captured_kwargs: dict[str, object] = {}

    class _DummyTokenizer:
        eos_token_id = 99
        _id_to_text = {
            11: "<prompt_a>",
            12: "<prompt_b>",
            21: "]}",
            22: "]},",
            23: "],",
            24: "]}\"",
            25: ",",
            26: " {\n",
            27: " {\"",
            28: " \"",
            29: "poly",
            30: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4',
            31: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]}',
            32: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]},\n     ',
            33: '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]},\n     {\n',
            34: (
                '{"objects": [{"desc": "book", "bbox_2d": [1, 2, 3, 4]},\n'
                '     {\n'
                '       "desc": "chair", "bbox_2d": [5, 6, 7, 8'
            ),
            99: "<|endoftext|>",
        }

        def decode(self, token_ids, **_kwargs) -> str:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return "".join(self._id_to_text[int(tok)] for tok in token_ids)

        def batch_decode(self, token_ids, **_kwargs):
            return [self.decode(ids, **_kwargs) for ids in token_ids]

        def get_vocab(self):
            return {f"tok_{tok_id}": tok_id for tok_id in self._id_to_text}

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
            logits_processor = kwargs.get("logits_processor")
            assert logits_processor is not None
            scores = torch.zeros((1, 100), dtype=torch.float)

            processed_bbox_tail = logits_processor(
                torch.tensor([[11, 12, 30]], dtype=torch.long),
                scores.clone(),
            )
            assert torch.isneginf(processed_bbox_tail[0, 21])
            assert processed_bbox_tail[0, 22].item() == pytest.approx(8.5)
            assert torch.isneginf(processed_bbox_tail[0, 23])
            assert torch.isneginf(processed_bbox_tail[0, 24])

            processed_followup = logits_processor(
                torch.tensor([[11, 12, 32]], dtype=torch.long),
                scores.clone(),
            )
            assert processed_followup[0, 26].item() == pytest.approx(8.5)
            assert processed_followup[0, 27].item() == pytest.approx(8.5)
            assert torch.isneginf(processed_followup[0, 28])
            assert processed_followup[0, 29].item() == pytest.approx(0.0)

            processed_after_open = logits_processor(
                torch.tensor([[11, 12, 33]], dtype=torch.long),
                scores.clone(),
            )
            assert processed_after_open[0, 26].item() == pytest.approx(0.0)
            assert processed_after_open[0, 27].item() == pytest.approx(0.0)
            assert not torch.isneginf(processed_after_open[0, 28])

            processed_second_bbox_tail = logits_processor(
                torch.tensor([[11, 12, 34]], dtype=torch.long),
                scores.clone(),
            )
            assert processed_second_bbox_tail[0, 21].item() == pytest.approx(0.0)
            assert processed_second_bbox_tail[0, 22].item() == pytest.approx(0.0)
            assert processed_second_bbox_tail[0, 23].item() == pytest.approx(0.0)
            assert processed_second_bbox_tail[0, 24].item() == pytest.approx(0.0)

            return torch.tensor([[11, 12, 30, 22]], dtype=torch.long)

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
            stop_pressure_mode="steer_bbox_tail_then_object_open_once",
            stop_pressure_min_new_tokens=0,
            stop_pressure_trigger_rule="raw_text_object_boundary",
            stop_pressure_logit_bias=8.5,
        ),
        _build_messages=lambda _image: [{"role": "user", "content": "describe"}],
    )

    results = generate_hf_batch(
        owner=owner,
        images=[image],
        result_factory=lambda **kwargs: kwargs,
    )

    assert len(results) == 1
    assert captured_kwargs["logits_processor"] is not None


def test_generate_hf_batch_does_not_suppress_bbox_closing_bracket_inside_object():
    image = Image.new("RGB", (8, 8), color=(0, 0, 0))

    class _DummyTokenizer:
        eos_token_id = 99
        _id_to_text = {
            11: "<prompt_a>",
            12: "<prompt_b>",
            21: "]",
            31: '{"objects": [{"desc": "book", "bbox_2d": [1, 2',
            99: "<|endoftext|>",
        }

        def decode(self, token_ids, **_kwargs) -> str:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return "".join(self._id_to_text[int(tok)] for tok in token_ids)

        def batch_decode(self, token_ids, **_kwargs):
            return [self.decode(ids, **_kwargs) for ids in token_ids]

        def get_vocab(self):
            return {f"tok_{tok_id}": tok_id for tok_id in self._id_to_text}

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
            logits_processor = kwargs.get("logits_processor")
            assert logits_processor is not None
            scores = torch.zeros((1, 100), dtype=torch.float)
            processed = logits_processor(
                torch.tensor([[11, 12, 31]], dtype=torch.long),
                scores.clone(),
            )
            assert not torch.isneginf(processed[0, 21])
            assert not torch.isneginf(processed[0, 99])
            return torch.tensor([[11, 12, 31, 21]], dtype=torch.long)

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
            stop_pressure_mode="suppress_terminating_tokens_after_object_boundary",
            stop_pressure_min_new_tokens=0,
            stop_pressure_trigger_rule="raw_text_object_boundary",
        ),
        _build_messages=lambda _image: [{"role": "user", "content": "describe"}],
    )

    results = generate_hf_batch(
        owner=owner,
        images=[image],
        result_factory=lambda **kwargs: kwargs,
    )

    assert len(results) == 1
