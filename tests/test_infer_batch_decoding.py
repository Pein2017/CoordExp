import json
from pathlib import Path

from PIL import Image

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
