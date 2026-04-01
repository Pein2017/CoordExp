import json
import sys
import types
from pathlib import Path

from PIL import Image

import src.infer.engine as infer_engine

from src.infer.engine import (
    GenerationConfig,
    GenerationResult,
    InferenceConfig,
    InferenceEngine,
)


def _write_img(path: Path, *, size: int = 32) -> None:
    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    img.save(path)


def test_infer_hf_batch_size_microbatches(tmp_path, monkeypatch):
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    # Arrange: 3 samples so batch_size=2 flushes [2, 1].
    for i in range(3):
        _write_img(tmp_path / f"img_{i}.png")

    gt_path = tmp_path / "gt.jsonl"
    with gt_path.open("w", encoding="utf-8") as f:
        for i in range(3):
            rec = {
                "images": [f"img_{i}.png"],
                "width": 32,
                "height": 32,
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": "obj"}],
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    out_path = tmp_path / "gt_vs_pred.jsonl"
    summary_path = tmp_path / "summary.json"

    inf_cfg = InferenceConfig(
        gt_jsonl=str(gt_path),
        model_checkpoint="dummy",
        mode="text",
        pred_coord_mode="auto",
        out_path=str(out_path),
        summary_path=str(summary_path),
        device="cpu",
        limit=0,
        backend_type="hf",
        backend={},
        detect_samples=1,
    )
    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=16,
        repetition_penalty=1.0,
        batch_size=2,
        seed=123,
    )

    engine = InferenceEngine(inf_cfg, gen_cfg)

    # Avoid loading a real HF model.
    monkeypatch.setattr(engine, "load_model", lambda: None)

    calls: list[int] = []

    def _fake_generate_batch(images):
        calls.append(len(images))
        text = '{"objects": [{"desc": "obj", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_10|>, <|coord_10|>]}]}<|im_end|>'
        return [GenerationResult(text=text, error=None) for _ in images]

    monkeypatch.setattr(engine, "_generate_batch", _fake_generate_batch)

    # Act
    got_out, got_summary = engine.infer()

    # Assert
    assert got_out == out_path
    assert got_summary == summary_path
    assert calls == [2, 1]

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    for line in lines:
        rec = json.loads(line)
        assert rec["errors"] == []
        assert rec["raw_output_json"] is not None
        assert rec["raw_ends_with_im_end"] is True
        assert len(rec["gt"]) == 1
        assert len(rec["pred"]) == 1


def test_infer_writes_pred_token_trace_sidecar(tmp_path, monkeypatch):
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    _write_img(tmp_path / "img_0.png")
    gt_path = tmp_path / "gt.jsonl"
    gt_path.write_text(
        json.dumps(
            {
                "images": ["img_0.png"],
                "width": 32,
                "height": 32,
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": "obj"}],
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "gt_vs_pred.jsonl"
    trace_path = tmp_path / "pred_token_trace.jsonl"
    summary_path = tmp_path / "summary.json"

    inf_cfg = InferenceConfig(
        gt_jsonl=str(gt_path),
        model_checkpoint="dummy",
        mode="text",
        pred_coord_mode="auto",
        out_path=str(out_path),
        pred_token_trace_path=str(trace_path),
        summary_path=str(summary_path),
        device="cpu",
        limit=0,
        backend_type="hf",
        backend={},
        detect_samples=1,
    )
    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=16,
        repetition_penalty=1.0,
        batch_size=1,
        seed=123,
    )

    engine = InferenceEngine(inf_cfg, gen_cfg)
    monkeypatch.setattr(engine, "load_model", lambda: None)

    def _fake_generate_batch(images):
        text = '{"objects": [{"desc": "obj", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_10|>, <|coord_10|>]}]}<|im_end|>'
        return [
            GenerationResult(
                text=text,
                generated_token_text=[
                    "<|coord_0|>",
                    "<|coord_0|>",
                    "<|coord_10|>",
                    "<|coord_10|>",
                    "<|im_end|>",
                ],
                token_logprobs=[-0.1, -0.1, -0.2, -0.2, -0.05],
                error=None,
            )
            for _ in images
        ]

    monkeypatch.setattr(engine, "_generate_batch", _fake_generate_batch)
    engine.infer()

    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(trace_rows) == 1
    assert trace_rows[0]["line_idx"] == 0
    assert trace_rows[0]["generated_token_text"] == [
        "<|coord_0|>",
        "<|coord_0|>",
        "<|coord_10|>",
        "<|coord_10|>",
        "<|im_end|>",
    ]
    assert trace_rows[0]["token_logprobs"] == [-0.1, -0.1, -0.2, -0.2, -0.05]

    rec = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert rec["raw_special_tokens"] == [
        "<|coord_0|>",
        "<|coord_0|>",
        "<|coord_10|>",
        "<|coord_10|>",
        "<|im_end|>",
    ]


def test_hf_attention_backend_fallback_is_recorded_in_summary(tmp_path, monkeypatch):
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    _write_img(tmp_path / "img_0.png")

    gt_path = tmp_path / "gt.jsonl"
    gt_path.write_text(
        json.dumps(
            {
                "images": ["img_0.png"],
                "width": 32,
                "height": 32,
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": "obj"}],
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "gt_vs_pred.jsonl"
    summary_path = tmp_path / "summary.json"

    inf_cfg = InferenceConfig(
        gt_jsonl=str(gt_path),
        model_checkpoint="dummy",
        mode="text",
        pred_coord_mode="auto",
        out_path=str(out_path),
        summary_path=str(summary_path),
        device="cpu",
        limit=0,
        backend_type="hf",
        backend={"attn_implementation": "flash_attention_2"},
        detect_samples=1,
    )
    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=16,
        repetition_penalty=1.0,
        batch_size=1,
        seed=123,
    )

    class _DummyModel:
        def to(self, _device: str):
            return self

        def eval(self):
            return self

    def _fake_from_pretrained(model_checkpoint: str, *, attn_implementation: str, **_kwargs):
        assert model_checkpoint == "dummy"
        if attn_implementation == "flash_attention_2":
            raise RuntimeError("flash attention not available")
        if attn_implementation == "sdpa":
            return _DummyModel()
        raise RuntimeError(f"unexpected attn_implementation={attn_implementation}")

    class _DummyTokenizer:
        padding_side = "right"
        pad_token_id = None
        eos_token_id = 1

    class _DummyProcessor:
        def __init__(self) -> None:
            self.tokenizer = _DummyTokenizer()

    class _DummyAutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _DummyProcessor()

    class _DummyQwen:
        @staticmethod
        def from_pretrained(model_checkpoint: str, **kwargs):
            return _fake_from_pretrained(model_checkpoint, **kwargs)

    monkeypatch.setattr(infer_engine, "AutoProcessor", _DummyAutoProcessor)
    monkeypatch.setattr(infer_engine, "Qwen3VLForConditionalGeneration", _DummyQwen)

    engine = InferenceEngine(inf_cfg, gen_cfg)
    engine.load_model()
    assert engine.attn_implementation_requested == "flash_attention_2"
    assert engine.attn_implementation_selected == "sdpa"

    def _fake_generate_batch(images):
        text = '{"objects": [{"desc": "obj", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_10|>, <|coord_10|>]}]}<|im_end|>'
        return [GenerationResult(text=text, error=None) for _ in images]

    monkeypatch.setattr(engine, "_generate_batch", _fake_generate_batch)

    _out, _summary = engine.infer()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["backend"]["attn_implementation_requested"] == "flash_attention_2"
    assert summary["backend"]["attn_implementation_selected"] == "sdpa"


def test_infer_emits_sample_scoped_errors_and_summary_counters(tmp_path, monkeypatch):
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    for i in range(2):
        _write_img(tmp_path / f"img_{i}.png")

    gt_path = tmp_path / "gt.jsonl"
    with gt_path.open("w", encoding="utf-8") as f:
        for i in range(2):
            rec = {
                "images": [f"img_{i}.png"],
                "width": 32,
                "height": 32,
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": "obj"}],
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    out_path = tmp_path / "gt_vs_pred.jsonl"
    summary_path = tmp_path / "summary.json"

    inf_cfg = InferenceConfig(
        gt_jsonl=str(gt_path),
        model_checkpoint="dummy",
        mode="text",
        pred_coord_mode="auto",
        out_path=str(out_path),
        summary_path=str(summary_path),
        device="cpu",
        limit=0,
        backend_type="hf",
        backend={},
        detect_samples=1,
    )
    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=16,
        repetition_penalty=1.0,
        batch_size=2,
        seed=123,
    )

    engine = InferenceEngine(inf_cfg, gen_cfg)

    # Avoid loading a real HF model.
    monkeypatch.setattr(engine, "load_model", lambda: None)
    bad = "not-json-output"

    def _fake_generate_batch(images):
        assert len(images) == 2
        ok = '{"objects": [{"desc": "obj", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_10|>, <|coord_10|>]}]}<|im_end|>'
        return [
            GenerationResult(text=ok, error=None),
            GenerationResult(text=bad, error=None),
        ]

    monkeypatch.setattr(engine, "_generate_batch", _fake_generate_batch)

    engine.infer()

    rows = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2

    assert rows[0]["errors"] == []
    assert rows[0]["pred"]

    assert "empty_pred" in rows[1]["errors"]
    assert rows[1]["pred"] == []
    assert rows[1]["raw_output_json"] is None
    assert rows[1]["error_entries"]
    assert rows[1]["error_entries"][0]["code"] == "empty_pred"
    assert rows[1]["error_entries"][0]["stage"] == "infer.parse_pred"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["errors_by_code"]["empty_pred"] == 1
    assert summary["errors_total"] == 1


def test_infer_summary_records_prompt_variant(tmp_path, monkeypatch):
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    _write_img(tmp_path / "img_0.png")

    gt_path = tmp_path / "gt.jsonl"
    gt_path.write_text(
        json.dumps(
            {
                "images": ["img_0.png"],
                "width": 32,
                "height": 32,
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": "obj"}],
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "gt_vs_pred.jsonl"
    summary_path = tmp_path / "summary.json"

    inf_cfg = InferenceConfig(
        gt_jsonl=str(gt_path),
        model_checkpoint="dummy",
        mode="text",
        prompt_variant="coco_80",
        object_field_order="geometry_first",
        object_ordering="random",
        pred_coord_mode="auto",
        out_path=str(out_path),
        summary_path=str(summary_path),
        device="cpu",
        limit=0,
        backend_type="hf",
        backend={},
        detect_samples=1,
    )
    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=16,
        repetition_penalty=1.0,
        batch_size=1,
        seed=123,
    )

    engine = InferenceEngine(inf_cfg, gen_cfg)
    monkeypatch.setattr(engine, "load_model", lambda: None)

    def _fake_generate_batch(images):
        text = '{"objects": [{"desc": "obj", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_10|>, <|coord_10|>]}]}<|im_end|>'
        return [GenerationResult(text=text, error=None) for _ in images]

    monkeypatch.setattr(engine, "_generate_batch", _fake_generate_batch)

    engine.infer()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["infer"]["prompt_variant"] == "coco_80"
    assert summary["infer"]["object_field_order"] == "geometry_first"
    assert summary["infer"]["object_ordering"] == "random"


def test_infer_preserves_image_id_and_metadata_in_gt_vs_pred(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    _write_img(tmp_path / "img_0.png")

    gt_path = tmp_path / "gt.jsonl"
    gt_path.write_text(
        json.dumps(
            {
                "images": ["img_0.png"],
                "image_id": 123,
                "width": 32,
                "height": 32,
                "metadata": {
                    "dataset": "lvis",
                    "dataset_policy": "lvis_federated",
                    "image_id": 123,
                    "lvis": {
                        "gt_objects": [
                            {"id": 1, "name": "cat", "frequency": "rare"}
                        ],
                        "positive_categories": [
                            {"id": 1, "name": "cat", "frequency": "rare"}
                        ],
                        "neg_categories": [
                            {"id": 2, "name": "dog", "frequency": "common"}
                        ],
                        "not_exhaustive_categories": [],
                    },
                },
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": "cat"}],
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "gt_vs_pred.jsonl"
    summary_path = tmp_path / "summary.json"

    inf_cfg = InferenceConfig(
        gt_jsonl=str(gt_path),
        model_checkpoint="dummy",
        mode="text",
        pred_coord_mode="auto",
        out_path=str(out_path),
        summary_path=str(summary_path),
        device="cpu",
        limit=0,
        backend_type="hf",
        backend={},
        detect_samples=1,
    )
    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=16,
        repetition_penalty=1.0,
        batch_size=1,
        seed=123,
    )

    engine = InferenceEngine(inf_cfg, gen_cfg)
    monkeypatch.setattr(engine, "load_model", lambda: None)

    def _fake_generate_batch(images):
        text = '{"objects": [{"desc": "cat", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_10|>, <|coord_10|>]}]}<|im_end|>'
        return [GenerationResult(text=text, error=None) for _ in images]

    monkeypatch.setattr(engine, "_generate_batch", _fake_generate_batch)
    engine.infer()

    rec = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert rec["image_id"] == 123
    assert rec["metadata"]["dataset_policy"] == "lvis_federated"
    assert rec["metadata"]["lvis"]["positive_categories"][0]["name"] == "cat"


def test_infer_build_messages_respects_random_ordering() -> None:
    engine = InferenceEngine(
        InferenceConfig(
            gt_jsonl="dummy.jsonl",
            model_checkpoint="dummy",
            mode="text",
            prompt_variant="coco_80",
            object_ordering="random",
        ),
        GenerationConfig(),
    )

    messages = engine._build_messages(Image.new("RGB", (16, 16), color=(0, 0, 0)))
    system_text = messages[0]["content"][0]["text"]
    user_text = messages[1]["content"][0]["text"]

    assert "any ordering is acceptable" in system_text
    assert "any ordering is acceptable" in user_text


def test_generate_vllm_server_preserves_coord_special_tokens_in_response_payload(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"objects":[{"desc":"obj","bbox_2d":[<|coord_1|>,<|coord_2|>,<|coord_3|>,<|coord_4|>]}]}'
                        }
                    }
                ]
            }

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse()

    fake_requests = types.SimpleNamespace(post=_fake_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    engine = InferenceEngine(
        InferenceConfig(
            gt_jsonl="dummy.jsonl",
            model_checkpoint="dummy-checkpoint",
            mode="text",
            prompt_variant="coco_80",
            object_field_order="desc_first",
            object_ordering="sorted",
            pred_coord_mode="auto",
            device="cpu",
            limit=0,
            backend_type="vllm",
            backend={"base_url": "http://127.0.0.1:8000", "timeout_s": 12.5},
            detect_samples=1,
        ),
        GenerationConfig(
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=32,
            repetition_penalty=1.05,
            batch_size=1,
            seed=42,
        ),
    )

    text = engine._generate_vllm_server(Image.new("RGB", (8, 8), color=(0, 0, 0)))

    assert "<|coord_1|>" in text
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["skip_special_tokens"] is False
    assert payload["spaces_between_special_tokens"] is False
    assert payload["stream"] is False
