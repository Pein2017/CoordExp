from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from src.infer.backend import DetectionDecodeResult
from src.infer.runtime import (
    GenerationResult,
    InferenceRuntime,
    make_offline_generation_config,
    make_offline_inference_config,
    run_offline_artifact_inference,
)


def test_inference_runtime_normalizes_legacy_backend_results() -> None:
    calls: list[tuple[object, list[object]]] = []

    def _fake_generate_batch(*, owner, images, result_factory):
        calls.append((owner, list(images)))
        return [
            result_factory(
                text="hello",
                generated_token_text=["h", "ello"],
                token_logprobs=[-0.1, -0.2],
                error=None,
            )
        ]

    owner = types.SimpleNamespace(cfg=types.SimpleNamespace(backend_type="hf"))
    runtime = InferenceRuntime(owner=owner, backend_generate=_fake_generate_batch)

    results = runtime.generate_many(images=[object()])

    assert calls == [(owner, results[0].backend_metadata["input_images"])]
    assert results == [
        DetectionDecodeResult(
            text="hello",
            generated_token_ids=None,
            generated_tokens=["h", "ello"],
            generated_logprobs=[-0.1, -0.2],
            stop_reason=None,
            backend="hf",
            backend_metadata={
                "response_family": "legacy_generation_result",
                "input_images": results[0].backend_metadata["input_images"],
            },
        )
    ]


def test_offline_engine_generate_batch_preserves_trace_fields(monkeypatch) -> None:
    import src.infer.backend as backend_module
    from src.infer.runtime import OfflineInferenceEngine

    def _fake_generate_batch(*, owner, images, result_factory):
        return [
            result_factory(
                text="ok",
                generated_token_ids=[101, 102],
                generated_token_text=["o", "k"],
                token_logprobs=[-0.1, -0.2],
                prompt_token_ids=[11, 12],
                stop_reason="stop",
                error=None,
            )
        ]

    monkeypatch.setattr(backend_module, "generate_batch", _fake_generate_batch)
    engine = OfflineInferenceEngine(
        make_offline_inference_config(
            gt_jsonl="dummy.jsonl",
            model_checkpoint="checkpoint",
            mode="text",
        ),
        make_offline_generation_config(),
    )

    [result] = engine._generate_batch([object()])

    assert result.text == "ok"
    assert result.generated_token_ids == [101, 102]
    assert result.generated_token_text == ["o", "k"]
    assert result.token_logprobs == [-0.1, -0.2]
    assert result.prompt_token_ids == [11, 12]
    assert result.stop_reason == "stop"


def test_inference_runtime_converts_backend_errors_to_result_errors() -> None:
    err = RuntimeError("boom")

    def _fake_generate_batch(*, owner, images, result_factory):
        return [result_factory(text="", error=err)]

    runtime = InferenceRuntime(
        owner=types.SimpleNamespace(cfg=types.SimpleNamespace(backend_type="vllm")),
        backend_generate=_fake_generate_batch,
    )

    [result] = runtime.generate_many(images=[object()])

    assert result.text == ""
    assert result.backend == "vllm"
    assert result.backend_metadata["error"] == "boom"


def test_inference_runtime_default_backend_does_not_import_legacy_engine(
    monkeypatch,
) -> None:
    import src.infer.backend as backend_module

    def _fake_generate_batch(*, owner, images, result_factory):
        return [result_factory(text=f"{owner.name}:{len(images)}", error=None)]

    monkeypatch.setitem(sys.modules, "src.infer.engine", None)
    monkeypatch.setattr(backend_module, "generate_batch", _fake_generate_batch)

    runtime = InferenceRuntime(
        owner=types.SimpleNamespace(
            name="owner",
            cfg=types.SimpleNamespace(backend_type="hf"),
        )
    )

    [result] = runtime.generate_many(images=[object()])

    assert result.text == "owner:1"
    assert result.backend == "hf"


def test_offline_config_factories_do_not_import_legacy_engine(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "src.infer.engine", None)

    from src.infer.runtime import (
        make_offline_generation_config,
        make_offline_generation_result,
        make_offline_inference_config,
        make_offline_run_counters,
    )

    gen_cfg = make_offline_generation_config(max_new_tokens=17)
    inf_cfg = make_offline_inference_config(
        gt_jsonl="gt.jsonl",
        model_checkpoint="checkpoint",
        mode="text",
    )
    result = make_offline_generation_result(text="ok", token_logprobs=[-0.25])
    counters = make_offline_run_counters()
    counters.add("invalid_json")

    assert type(gen_cfg).__module__ == "src.infer.runtime"
    assert gen_cfg.max_new_tokens == 17
    assert type(inf_cfg).__module__ == "src.infer.runtime"
    assert inf_cfg.gt_jsonl == "gt.jsonl"
    assert type(result).__module__ == "src.infer.runtime"
    assert result.token_logprobs == [-0.25]
    assert type(counters).__module__ == "src.infer.runtime"
    assert counters.to_summary()["errors_by_code"] == {"invalid_json": 1}


def test_runtime_import_stays_light_without_legacy_engine_or_backend_deps() -> None:
    code = r'''
import importlib
import json
import sys

blocked = {"src.infer.engine", "torch", "PIL", "transformers", "vllm", "requests"}
for name in list(sys.modules):
    if name in blocked or any(name.startswith(prefix + ".") for prefix in blocked):
        sys.modules.pop(name, None)

importlib.import_module("src.infer.runtime")

loaded = sorted(
    name
    for name in sys.modules
    if name in blocked or any(name.startswith(prefix + ".") for prefix in blocked)
)
print(json.dumps(loaded))
if loaded:
    raise SystemExit(1)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_run_offline_inference_uses_runtime_artifact_runner(monkeypatch) -> None:
    import src.infer.runtime as runtime_module

    class _FakeEngine:
        def __init__(self, cfg, gen_cfg, *, logger=None) -> None:
            self.cfg = cfg
            self.gen_cfg = gen_cfg
            self.logger = logger
            self.model = None
            self.processor = None

        def infer(self):  # pragma: no cover - this must not be called
            raise AssertionError("run_offline_inference must call runtime runner")

    captured: dict[str, object] = {}

    def _fake_runner(owner):
        captured["owner"] = owner
        captured["cfg"] = owner.cfg
        captured["gen_cfg"] = owner.gen_cfg
        captured["model"] = owner.model
        captured["processor"] = owner.processor
        return "gt_vs_pred.jsonl", "summary.json"

    model = object()
    processor = object()
    monkeypatch.setattr(runtime_module, "OfflineInferenceEngine", _FakeEngine)
    monkeypatch.setattr(runtime_module, "run_offline_artifact_inference", _fake_runner)

    result = runtime_module.run_offline_inference(
        inference_kwargs={
            "gt_jsonl": "gt.jsonl",
            "model_checkpoint": "checkpoint",
            "mode": "text",
        },
        generation_kwargs={"max_new_tokens": 23},
        model=model,
        processor=processor,
        logger="logger",
    )

    assert captured["cfg"].gt_jsonl == "gt.jsonl"
    assert captured["gen_cfg"].max_new_tokens == 23
    assert captured["model"] is model
    assert captured["processor"] is processor
    assert result.base_jsonl_path == "gt_vs_pred.jsonl"
    assert result.summary_path == "summary.json"
    assert result.processor is processor


class _FakeOfflineLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple[object, ...]]] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append((message, args))


class _FakeOfflineCoord:
    def __init__(self, owner: "_FakeOfflineOwner") -> None:
        self.owner = owner

    def process_record_gt(self, record, *, width: int, height: int, errors: list[str]):
        self.owner.calls.append(
            "process_gt_loaded" if self.owner.model_loaded else "process_gt_preload"
        )
        return [{"type": "bbox_2d", "points": [0, 0, width, height], "desc": "gt"}]

    def process_prediction_text(
        self,
        raw_text: str,
        *,
        width: int,
        height: int,
        errors: list[str],
    ):
        self.owner.calls.append("process_pred")
        return [{"type": "bbox_2d", "points": [0, 0, 1, 1], "desc": "pred"}]


class _FakeOfflineOwner:
    def __init__(self, tmp_path: Path) -> None:
        self.calls: list[str] = []
        self.model_loaded = False
        self.logger = _FakeOfflineLogger()
        self.resolved_mode = "text"
        self.requested_mode = "text"
        self.mode_reason = None
        self.prompt_variant = "default"
        self.bbox_format = "xyxy"
        self.detection_sequence_format = "coordjson"
        self.object_field_order = "desc_first"
        self.object_ordering = "sorted"
        self.prompt_template_hash = "prompt-hash"
        self.attn_implementation_requested = "sdpa"
        self.attn_implementation_selected = "sdpa"
        self.processor = None
        self.tokenizer = None
        self.qwen_generation_token_ids = None
        self.coord = _FakeOfflineCoord(self)
        self.cfg = make_offline_inference_config(
            gt_jsonl=str(tmp_path / "gt.jsonl"),
            model_checkpoint="checkpoint",
            mode="text",
            pred_coord_mode="auto",
            out_path=str(tmp_path / "gt_vs_pred.jsonl"),
            pred_token_trace_path=str(tmp_path / "pred_token_trace.jsonl"),
            summary_path=str(tmp_path / "summary.json"),
            device="cpu",
            limit=0,
            backend_type="hf",
            backend={},
            prompt_policy_fingerprint="prompt_policy:test",
            decode_policy_fingerprint="decode:test",
            model_identity_fingerprint="model:test",
        )
        self.gen_cfg = make_offline_generation_config(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=8,
            repetition_penalty=1.0,
            batch_size=1,
            seed=7,
        )

    def load_model(self) -> None:
        self.calls.append("load_model")
        self.model_loaded = True

    def _generate_batch(self, images):
        self.calls.append("generate_batch")
        return [
            GenerationResult(
                text='{"objects":[{"desc":"pred","bbox_2d":[0,0,1,1]}]}<|im_end|>',
                generated_token_text=["{", "}"],
                token_logprobs=[-0.1, -0.2],
            )
            for _image in images
        ]


def _write_runner_gt(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (32, 24), color=(128, 128, 128)).save(path.parent / "img.png")
    path.write_text(
        json.dumps(
            {
                "images": ["img.png"],
                "width": 32,
                "height": 24,
                "image_id": 17,
                "metadata": {"split": "tiny"},
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": "gt"}],
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_runtime_artifact_runner_emits_golden_row_trace_and_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_runner_gt(tmp_path / "gt.jsonl")
    owner = _FakeOfflineOwner(tmp_path)
    monkeypatch.setitem(sys.modules, "src.infer.engine", None)
    for legacy_helper in (
        "_resolve_image_path",
        "_process_gt",
        "_compact_objects",
        "_prepare_image",
        "_process_pred",
    ):
        assert not hasattr(owner, legacy_helper)

    out_path, summary_path = run_offline_artifact_inference(owner)

    assert out_path == tmp_path / "gt_vs_pred.jsonl"
    assert summary_path == tmp_path / "summary.json"
    assert owner.calls.index("process_gt_preload") < owner.calls.index("load_model")
    assert owner.calls.index("load_model") < owner.calls.index("process_gt_loaded")
    assert "generate_batch" in owner.calls
    row = json.loads((tmp_path / "gt_vs_pred.jsonl").read_text(encoding="utf-8"))
    assert set(row) >= {
        "image",
        "width",
        "height",
        "mode",
        "coord_mode",
        "gt",
        "pred",
        "raw_output_json",
        "raw_special_tokens",
        "raw_ends_with_im_end",
        "errors",
        "error_entries",
        "image_id",
        "metadata",
    }
    assert row["image"] == "img.png"
    assert row["image_id"] == 17
    assert row["metadata"] == {"split": "tiny"}
    assert row["errors"] == []
    assert row["raw_ends_with_im_end"] is True
    assert row["pred"] == [{"type": "bbox_2d", "points": [0, 0, 1, 1], "desc": "pred"}]

    trace = json.loads(
        (tmp_path / "pred_token_trace.jsonl").read_text(encoding="utf-8")
    )
    assert trace == {
        "line_idx": 0,
        "generated_token_text": ["{", "}"],
        "token_logprobs": [-0.1, -0.2],
    }

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_read"] == 1
    assert summary["total_emitted"] == 1
    assert summary["errors_total"] == 0
    assert summary["backend"]["type"] == "hf"
    assert summary["infer"]["gt_jsonl"] == str(tmp_path / "gt.jsonl")


def test_runtime_artifact_runner_preflight_failure_writes_no_artifacts(
    tmp_path: Path,
) -> None:
    (tmp_path / "gt.jsonl").write_text("not json\n", encoding="utf-8")
    owner = _FakeOfflineOwner(tmp_path)

    with pytest.raises(ValueError, match="Inference preflight failed"):
        run_offline_artifact_inference(owner)

    assert owner.calls == []
    assert not (tmp_path / "gt_vs_pred.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "pred_token_trace.jsonl").exists()


def test_runtime_artifact_runner_rejects_image_size_mismatch_before_side_effects(
    tmp_path: Path,
) -> None:
    from PIL import Image

    Image.new("RGB", (2, 2), color=(128, 128, 128)).save(tmp_path / "img.png")
    (tmp_path / "gt.jsonl").write_text(
        json.dumps(
            {
                "images": ["img.png"],
                "width": 32,
                "height": 24,
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": "gt"}],
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    owner = _FakeOfflineOwner(tmp_path)

    with pytest.raises(ValueError, match="Image size does not match"):
        run_offline_artifact_inference(owner)

    assert owner.calls == []
    assert not (tmp_path / "gt_vs_pred.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "pred_token_trace.jsonl").exists()


def test_runtime_artifact_runner_rejects_post_preflight_image_load_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import src.infer.runtime as runtime

    _write_runner_gt(tmp_path / "gt.jsonl")
    owner = _FakeOfflineOwner(tmp_path)

    def _missing_image(*_args, **_kwargs):
        return tmp_path / "img.png", None

    monkeypatch.setattr(runtime, "prepare_offline_image", _missing_image)

    with pytest.raises(ValueError, match="Failed to load image for inference"):
        run_offline_artifact_inference(owner)

    assert owner.calls == ["process_gt_preload", "load_model", "process_gt_loaded"]
    assert (tmp_path / "gt_vs_pred.jsonl").read_text(encoding="utf-8") == ""
    assert not (tmp_path / "summary.json").exists()
    assert (tmp_path / "pred_token_trace.jsonl").read_text(encoding="utf-8") == ""
