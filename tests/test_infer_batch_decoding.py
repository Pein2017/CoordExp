import json
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
        text = '{"0": {"desc": "obj", "bbox_2d": [0, 0, 10, 10]}}<|im_end|>'
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
        text = '{"0": {"desc": "obj", "bbox_2d": [0, 0, 10, 10]}}<|im_end|>'
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

    def _fake_generate_batch(images):
        assert len(images) == 2
        ok = '{"0": {"desc": "obj", "bbox_2d": [0, 0, 10, 10]}}<|im_end|>'
        bad = "not json"
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
        text = '{"0": {"desc": "obj", "bbox_2d": [0, 0, 10, 10]}}<|im_end|>'
        return [GenerationResult(text=text, error=None) for _ in images]

    monkeypatch.setattr(engine, "_generate_batch", _fake_generate_batch)

    engine.infer()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["infer"]["prompt_variant"] == "coco_80"
    assert summary["infer"]["object_field_order"] == "geometry_first"
