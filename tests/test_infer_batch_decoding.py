import hashlib
import json
import sys
import types
from pathlib import Path

from PIL import Image
import pytest
import torch

from src.infer.runtime import (
    OfflineInferenceEngine,
    make_offline_generation_config,
    make_offline_generation_result,
    make_offline_inference_config,
    make_offline_run_counters,
)
import src.infer.runtime as infer_runtime
from src.infer.backend import generate_hf_batch, generate_vllm_batch, generate_vllm_server_result
from src.infer.prompt import build_offline_detection_chat_messages

GenerationConfig = make_offline_generation_config
GenerationResult = make_offline_generation_result
InferenceConfig = make_offline_inference_config
InferenceEngine = OfflineInferenceEngine


def _write_img(path: Path, *, size: int = 32) -> None:
    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    img.save(path)


class _QwenSpecialTokenMixin:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<|endoftext|>": 0,
            "<|im_end|>": 1,
        }.get(token, self.unk_token_id)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        token_id = self.convert_tokens_to_ids(text)
        return [] if token_id == self.unk_token_id else [token_id]


def _write_adapter_checkpoint(
    path: Path,
    *,
    base_model_name_or_path: str = "base-model",
    with_coord_offset: bool = False,
    tie_head: bool = True,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    modules_to_save = ["coord_offset_adapter"] if with_coord_offset else []
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model_name_or_path,
                "modules_to_save": modules_to_save,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    if with_coord_offset:
        import torch
        from safetensors.torch import save_file

        payload = {
            "base_model.model.coord_offset_adapter.coord_ids": torch.tensor(
                [2, 5], dtype=torch.long
            ),
            "base_model.model.coord_offset_adapter.embed_offset": torch.zeros(
                2, 4, dtype=torch.float32
            ),
        }
        if not tie_head:
            payload["base_model.model.coord_offset_adapter.head_offset"] = (
                torch.zeros(2, 4, dtype=torch.float32)
            )
        save_file(payload, str(path / "adapter_model.safetensors"))


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
        allow_diagnostic_gt_vs_pred=True,
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
        allow_diagnostic_gt_vs_pred=True,
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
    expected_raw_text = '{"objects": [{"desc": "obj", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_10|>, <|coord_10|>]}]}<|im_end|>'
    assert trace_rows[0]["raw_output_sha256"] == hashlib.sha256(
        expected_raw_text.encode("utf-8")
    ).hexdigest()
    expected_trace_payload = {
        "generated_token_text": [
            "<|coord_0|>",
            "<|coord_0|>",
            "<|coord_10|>",
            "<|coord_10|>",
            "<|im_end|>",
        ],
        "token_logprobs": [-0.1, -0.1, -0.2, -0.2, -0.05],
    }
    assert trace_rows[0]["token_trace_sha256"] == hashlib.sha256(
        json.dumps(
            expected_trace_payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    rec = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert rec["raw_special_tokens"] == [
        "<|coord_0|>",
        "<|coord_0|>",
        "<|coord_10|>",
        "<|coord_10|>",
        "<|im_end|>",
    ]


def test_hf_batch_compact_grammar_uses_padded_prompt_offset(monkeypatch):
    captured: dict[str, list[int]] = {}

    class _DummyTokenizer(_QwenSpecialTokenMixin):
        padding_side = "left"
        pad_token_id = 0
        eos_token_id = 1

        def batch_decode(self, token_ids, **_kwargs):
            return [self.decode(ids, **_kwargs) for ids in token_ids]

        def decode(self, token_ids, **_kwargs):
            ids = [int(value) for value in token_ids]
            return "".join("<|im_end|>" if value == 1 else "x" for value in ids)

    class _DummyProcessor:
        def __init__(self) -> None:
            self.tokenizer = _DummyTokenizer()

        def apply_chat_template(self, _message, *, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is False
            return "prompt"

        def __call__(self, **_kwargs):
            return {
                "input_ids": torch.tensor(
                    [
                        [0, 0, 10, 11],
                        [20, 21, 22, 23],
                    ],
                    dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    [
                        [0, 0, 1, 1],
                        [1, 1, 1, 1],
                    ],
                    dtype=torch.long,
                ),
            }

    class _DummyGenerateOutput:
        def __init__(self) -> None:
            self.sequences = torch.tensor(
                [
                    [0, 0, 10, 11, 1],
                    [20, 21, 22, 23, 1],
                ],
                dtype=torch.long,
            )
            self.scores = [torch.zeros((2, 32), dtype=torch.float32)]

    class _DummyModel:
        def generate(self, **_kwargs):
            return _DummyGenerateOutput()

    def _fake_build_compact_grammar_logits_processor(
        *, tokenizer, prompt_lengths, detection_sequence_format, force_row_start
    ):
        captured["prompt_lengths"] = list(prompt_lengths)
        assert detection_sequence_format == "compact_full"
        assert force_row_start is True

        def _processor(input_ids, scores):
            return scores

        return _processor

    import src.infer.constraints as constraints

    monkeypatch.setattr(
        constraints,
        "build_compact_grammar_logits_processor",
        _fake_build_compact_grammar_logits_processor,
    )

    owner = types.SimpleNamespace(
        model=_DummyModel(),
        processor=_DummyProcessor(),
        cfg=types.SimpleNamespace(device="cpu"),
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=1,
            repetition_penalty=1.0,
            batch_size=2,
            seed=123,
            compact_grammar_enabled=True,
            compact_grammar_format="compact_full",
            compact_grammar_force_row_start=True,
        ),
        system_prompt="system",
        user_prompt="prompt",
    )

    results = generate_hf_batch(
        owner=owner,
        images=[
            Image.new("RGB", (8, 8), color=(0, 0, 0)),
            Image.new("RGB", (8, 8), color=(0, 0, 0)),
        ],
        result_factory=GenerationResult,
    )

    assert captured["prompt_lengths"] == [4, 4]
    assert [result.text for result in results] == ["<|im_end|>", "<|im_end|>"]


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

    class _DummyTokenizer(_QwenSpecialTokenMixin):
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

    monkeypatch.setattr(infer_runtime, "AutoProcessor", _DummyAutoProcessor)
    monkeypatch.setattr(infer_runtime, "Qwen3VLForConditionalGeneration", _DummyQwen)

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


def test_hf_adapter_checkpoint_loads_via_swift_shorthand_and_records_resolved_base(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    _write_img(tmp_path / "img_0.png")
    adapter_dir = tmp_path / "adapter-dir"
    _write_adapter_checkpoint(
        adapter_dir,
        base_model_name_or_path="base-model",
    )

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
        model_checkpoint=str(adapter_dir),
        mode="text",
        pred_coord_mode="auto",
        out_path=str(out_path),
        summary_path=str(summary_path),
        device="cpu",
        limit=0,
        backend_type="hf",
        backend={},
        detect_samples=1,
        allow_diagnostic_gt_vs_pred=True,
    )
    gen_cfg = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=16,
        repetition_penalty=1.0,
        batch_size=1,
        seed=123,
    )

    load_calls: dict[str, list[object]] = {"qwen": [], "processor": [], "swift": []}

    class _DummyBaseModel:
        def __init__(self) -> None:
            self.device = None

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            return self

    class _WrappedModel:
        def __init__(self, base_model) -> None:
            self.base_model = base_model
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return self

    class _DummyTokenizer(_QwenSpecialTokenMixin):
        padding_side = "right"
        pad_token_id = None
        eos_token_id = 1

    class _DummyProcessor:
        def __init__(self) -> None:
            self.tokenizer = _DummyTokenizer()

    class _DummyAutoProcessor:
        @staticmethod
        def from_pretrained(model_checkpoint: str, **_kwargs):
            load_calls["processor"].append(model_checkpoint)
            return _DummyProcessor()

    class _DummyQwen:
        @staticmethod
        def from_pretrained(model_checkpoint: str, **kwargs):
            load_calls["qwen"].append((model_checkpoint, kwargs["attn_implementation"]))
            return _DummyBaseModel()

    class _DummySwift:
        @staticmethod
        def from_pretrained(model, *, model_id: str, inference_mode: bool, **_kwargs):
            load_calls["swift"].append((model, model_id, inference_mode))
            return _WrappedModel(model)

    fake_swift_module = types.ModuleType("swift")
    fake_swift_module.Swift = _DummySwift

    monkeypatch.setattr(infer_runtime, "AutoProcessor", _DummyAutoProcessor)
    monkeypatch.setattr(infer_runtime, "Qwen3VLForConditionalGeneration", _DummyQwen)
    monkeypatch.setitem(sys.modules, "swift", fake_swift_module)

    engine = InferenceEngine(inf_cfg, gen_cfg)
    engine.load_model()

    assert load_calls["qwen"] == [("base-model", "sdpa")]
    assert load_calls["processor"] == ["base-model"]
    assert len(load_calls["swift"]) == 1
    assert load_calls["swift"][0][1:] == (str(adapter_dir), True)
    assert isinstance(engine.model, _WrappedModel)
    assert engine.model.eval_called is True

    def _fake_generate_batch(images):
        text = '{"objects": [{"desc": "obj", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_10|>, <|coord_10|>]}]}<|im_end|>'
        return [GenerationResult(text=text, error=None) for _ in images]

    monkeypatch.setattr(engine, "_generate_batch", _fake_generate_batch)
    engine.infer()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["backend"]["model_checkpoint"] == str(adapter_dir)
    assert summary["backend"]["adapter_checkpoint"] is None
    assert summary["backend"]["checkpoint_mode"] == "adapter_shorthand"
    assert summary["backend"]["requested_model_checkpoint"] == str(adapter_dir)
    assert summary["backend"]["resolved_base_model_checkpoint"] == "base-model"
    assert summary["backend"]["resolved_adapter_checkpoint"] == str(adapter_dir)


def test_hf_coord_offset_adapter_is_preinstalled_before_swift_reload(
    tmp_path, monkeypatch
):
    adapter_dir = tmp_path / "adapter-dir"
    _write_adapter_checkpoint(
        adapter_dir,
        base_model_name_or_path="base-model",
        with_coord_offset=True,
        tie_head=False,
    )

    inf_cfg = InferenceConfig(
        gt_jsonl=str(tmp_path / "gt.jsonl"),
        model_checkpoint=str(adapter_dir),
        mode="text",
        pred_coord_mode="auto",
        out_path=str(tmp_path / "gt_vs_pred.jsonl"),
        summary_path=str(tmp_path / "summary.json"),
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

    load_order: list[tuple[object, ...]] = []

    class _DummyBaseModel:
        def to(self, _device: str):
            return self

        def eval(self):
            return self

    class _WrappedModel:
        def __init__(self, base_model) -> None:
            self.base_model = base_model
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return self

    class _DummyTokenizer(_QwenSpecialTokenMixin):
        padding_side = "right"
        pad_token_id = None
        eos_token_id = 1

    class _DummyProcessor:
        def __init__(self) -> None:
            self.tokenizer = _DummyTokenizer()

    class _DummyAutoProcessor:
        @staticmethod
        def from_pretrained(model_checkpoint: str, **_kwargs):
            assert model_checkpoint == "base-model"
            return _DummyProcessor()

    class _DummyQwen:
        @staticmethod
        def from_pretrained(model_checkpoint: str, **kwargs):
            assert model_checkpoint == "base-model"
            assert kwargs["attn_implementation"] == "sdpa"
            return _DummyBaseModel()

    class _DummySwift:
        @staticmethod
        def from_pretrained(model, *, model_id: str, inference_mode: bool, **_kwargs):
            load_order.append(("swift", model_id, inference_mode))
            return _WrappedModel(model)

    def _fake_install(model, *, coord_ids, tie_head, dtype=None):
        load_order.append(("install", tuple(coord_ids), tie_head, dtype))
        return object()

    def _fake_reattach(model):
        load_order.append(("reattach", type(model).__name__))
        return object()

    fake_swift_module = types.ModuleType("swift")
    fake_swift_module.Swift = _DummySwift

    monkeypatch.setattr(infer_runtime, "AutoProcessor", _DummyAutoProcessor)
    monkeypatch.setattr(infer_runtime, "Qwen3VLForConditionalGeneration", _DummyQwen)
    monkeypatch.setattr(infer_runtime, "install_coord_offset_adapter", _fake_install)
    monkeypatch.setattr(infer_runtime, "reattach_coord_offset_hooks", _fake_reattach)
    monkeypatch.setitem(sys.modules, "swift", fake_swift_module)

    engine = InferenceEngine(inf_cfg, gen_cfg)
    engine.load_model()

    assert load_order == [
        ("install", (2, 5), False, None),
        ("swift", str(adapter_dir), True),
        ("reattach", "_WrappedModel"),
    ]
    assert isinstance(engine.model, _WrappedModel)
    assert engine.model.eval_called is True


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
        allow_diagnostic_gt_vs_pred=True,
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
    assert rows[1]["metric_bearing"] is False
    assert rows[1]["parser_policy"] == "diagnostic"
    assert rows[1]["error_entries"]
    assert rows[1]["error_entries"][0]["code"] == "empty_pred"
    assert rows[1]["error_entries"][0]["stage"] == "infer.parse_pred"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["errors_by_code"]["empty_pred"] == 1
    assert summary["errors_total"] == 1


def test_infer_rejects_diagnostic_gt_vs_pred_by_default(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        engine,
        "_generate_batch",
        lambda images: [GenerationResult(text="not-json-output", error=None)],
    )

    with pytest.raises(ValueError, match="metric_bearing=false"):
        engine.infer()


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

    messages = build_offline_detection_chat_messages(
        system_prompt=engine.system_prompt,
        user_prompt=engine.user_prompt,
        image=Image.new("RGB", (16, 16), color=(0, 0, 0)),
    )
    system_text = str(messages[0]["content"])
    user_content = messages[1]["content"]
    user_text = next(
        item["text"]
        for item in user_content
        if isinstance(item, dict) and item.get("type") == "text"
    )

    assert "any ordering is acceptable" in system_text
    assert "any ordering is acceptable" in user_text
    assert [item["type"] for item in user_content] == ["image", "text"]


def test_backend_generate_vllm_server_preserves_coord_special_tokens_in_response_payload(
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

    owner = types.SimpleNamespace(
        cfg=InferenceConfig(
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
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=32,
            repetition_penalty=1.05,
            batch_size=1,
            seed=42,
        ),
        system_prompt="system",
        user_prompt="detect",
    )

    result = generate_vllm_server_result(
        owner=owner,
        image=Image.new("RGB", (8, 8), color=(0, 0, 0)),
        result_factory=GenerationResult,
    )

    assert "<|coord_1|>" in result.text
    assert not hasattr(owner, "_generate_vllm_server_result")
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["skip_special_tokens"] is False
    assert payload["spaces_between_special_tokens"] is False
    assert payload["stream"] is False
    assert payload["stop"] == ["<|im_end|>"]
    content = payload["messages"][1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "detect"}


def test_backend_generate_vllm_server_trace_returns_strict_result_object(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "prompt_token_ids": [11, 12],
                "choices": [
                    {
                        "message": {"content": "ab"},
                        "finish_reason": "stop",
                        "token_ids": [101, 102],
                        "logprobs": {
                            "content": [
                                {"token": "a", "logprob": -0.1},
                                {"token": "b", "logprob": -0.2},
                            ]
                        },
                    }
                ],
            }

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return _FakeResponse()

    fake_requests = types.SimpleNamespace(post=_fake_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    owner = types.SimpleNamespace(
        cfg=InferenceConfig(
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
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=32,
            repetition_penalty=1.05,
            batch_size=1,
            seed=42,
            trace_logprobs=True,
        ),
        system_prompt="system",
        user_prompt="detect",
    )

    result = generate_vllm_server_result(
        owner=owner,
        image=Image.new("RGB", (8, 8), color=(0, 0, 0)),
        result_factory=GenerationResult,
    )

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["logprobs"] is True
    assert payload["return_token_ids"] is True
    assert payload["return_tokens_as_token_ids"] is True
    assert result.text == "ab"
    assert result.generated_token_ids == [101, 102]
    assert result.generated_token_text == ["a", "b"]
    assert result.token_logprobs == [-0.1, -0.2]
    assert result.prompt_token_ids == [11, 12]


def test_vllm_local_contract_sets_im_end_stop_and_disables_resize(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

        def chat(self, msg_batch, *, sampling_params, use_tqdm):
            captured["msg_batch"] = msg_batch
            captured["sampling_params"] = sampling_params
            captured["use_tqdm"] = use_tqdm
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="local-output")]
                )
                for _message in msg_batch
            ]

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_kwargs"] = kwargs

    fake_vllm = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    owner = types.SimpleNamespace(
        cfg=InferenceConfig(
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
            backend={
                "mode": "local",
                "model": "dummy-vllm-model",
                "server_options": {
                    "vllm_tensor_parallel_size": 1,
                    "vllm_max_model_len": 4096,
                },
            },
            detect_samples=1,
        ),
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=32,
            repetition_penalty=1.05,
            batch_size=1,
            seed=42,
        ),
        system_prompt="system",
        user_prompt="detect",
        vllm_llm=None,
    )

    results = generate_vllm_batch(
        owner=owner,
        images=[Image.new("RGB", (8, 8), color=(0, 0, 0))],
        result_factory=GenerationResult,
    )

    llm_kwargs = captured["llm_kwargs"]
    sampling_kwargs = captured["sampling_kwargs"]
    assert isinstance(llm_kwargs, dict)
    assert isinstance(sampling_kwargs, dict)
    assert llm_kwargs["model"] == "dummy-vllm-model"
    assert llm_kwargs["trust_remote_code"] is True
    assert llm_kwargs["allowed_local_media_path"] == str(Path(".").resolve())
    assert llm_kwargs["seed"] == 42
    assert llm_kwargs["tensor_parallel_size"] == 1
    assert llm_kwargs["max_model_len"] == 4096
    assert llm_kwargs["mm_processor_kwargs"] == {"do_resize": False}
    assert sampling_kwargs["stop"] == ["<|im_end|>"]
    assert "stop_token_ids" not in sampling_kwargs
    assert results[0].text == "local-output"
    assert not hasattr(owner, "_vllm_mode")
    assert not hasattr(owner, "_generate_vllm_local_batch")
    content = captured["msg_batch"][0][1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "detect"}


def test_vllm_local_trace_returns_strict_result_object(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

        def chat(self, msg_batch, *, sampling_params, use_tqdm):
            captured["msg_batch"] = msg_batch
            captured["sampling_params"] = sampling_params
            captured["use_tqdm"] = use_tqdm
            return [
                types.SimpleNamespace(
                    prompt_token_ids=[11, 12],
                    outputs=[
                        types.SimpleNamespace(
                            text="ab",
                            token_ids=[101, 102],
                            logprobs=[
                                {
                                    101: types.SimpleNamespace(
                                        logprob=-0.1,
                                        decoded_token="a",
                                    )
                                },
                                {
                                    102: types.SimpleNamespace(
                                        logprob=-0.2,
                                        decoded_token="b",
                                    )
                                },
                            ],
                            finish_reason="stop",
                        )
                    ],
                )
            ]

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_kwargs"] = kwargs

    fake_vllm = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    owner = types.SimpleNamespace(
        cfg=InferenceConfig(
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
            backend={"mode": "local", "model": "dummy-vllm-model"},
            detect_samples=1,
        ),
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=32,
            repetition_penalty=1.05,
            batch_size=1,
            seed=42,
            trace_logprobs=True,
        ),
        system_prompt="system",
        user_prompt="detect",
        vllm_llm=None,
    )

    [result] = generate_vllm_batch(
        owner=owner,
        images=[Image.new("RGB", (8, 8), color=(0, 0, 0))],
        result_factory=GenerationResult,
    )

    assert captured["sampling_kwargs"]["logprobs"] == 1
    assert result.text == "ab"
    assert result.generated_token_ids == [101, 102]
    assert result.generated_token_text == ["a", "b"]
    assert result.token_logprobs == [-0.1, -0.2]
    assert result.prompt_token_ids == [11, 12]
    assert result.stop_reason == "stop"


def test_vllm_local_trace_fails_when_logprobs_are_missing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

        def chat(self, msg_batch, *, sampling_params, use_tqdm):
            captured["sampling_params"] = sampling_params
            return [
                types.SimpleNamespace(
                    outputs=[
                        types.SimpleNamespace(
                            text="ab",
                            token_ids=[101, 102],
                            logprobs=None,
                        )
                    ],
                )
            ]

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_kwargs"] = kwargs

    fake_vllm = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    owner = types.SimpleNamespace(
        cfg=InferenceConfig(
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
            backend={"mode": "local", "model": "dummy-vllm-model"},
            detect_samples=1,
        ),
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=32,
            repetition_penalty=1.05,
            batch_size=1,
            seed=42,
            trace_logprobs=True,
        ),
        system_prompt="system",
        user_prompt="detect",
        vllm_llm=None,
    )

    with pytest.raises(RuntimeError, match="Missing vLLM local logprobs"):
        generate_vllm_batch(
            owner=owner,
            images=[Image.new("RGB", (8, 8), color=(0, 0, 0))],
            result_factory=GenerationResult,
        )

    assert captured["sampling_kwargs"]["logprobs"] == 1


def test_vllm_local_trace_fails_when_logprob_token_id_mismatches(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

        def chat(self, msg_batch, *, sampling_params, use_tqdm):
            captured["sampling_params"] = sampling_params
            return [
                types.SimpleNamespace(
                    outputs=[
                        types.SimpleNamespace(
                            text="a",
                            token_ids=[101],
                            logprobs=[
                                {
                                    999: types.SimpleNamespace(
                                        logprob=-0.1,
                                        decoded_token="wrong",
                                    )
                                }
                            ],
                        )
                    ],
                )
            ]

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_kwargs"] = kwargs

    fake_vllm = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    owner = types.SimpleNamespace(
        cfg=InferenceConfig(
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
            backend={"mode": "local", "model": "dummy-vllm-model"},
            detect_samples=1,
        ),
        gen_cfg=GenerationConfig(
            temperature=0.0,
            top_p=0.9,
            max_new_tokens=32,
            repetition_penalty=1.05,
            batch_size=1,
            seed=42,
            trace_logprobs=True,
        ),
        system_prompt="system",
        user_prompt="detect",
        vllm_llm=None,
    )

    with pytest.raises(RuntimeError, match="missing chosen token logprob"):
        generate_vllm_batch(
            owner=owner,
            images=[Image.new("RGB", (8, 8), color=(0, 0, 0))],
            result_factory=GenerationResult,
        )

    assert captured["sampling_kwargs"]["logprobs"] == 1


def test_hf_load_model_sets_missing_pad_to_endoftext_not_im_end(monkeypatch) -> None:
    class _FakeTokenizer:
        eos_token_id = 1
        pad_token_id = None
        unk_token_id = -1

        def convert_tokens_to_ids(self, token: str) -> int:
            return {
                "<|im_end|>": 1,
                "<|endoftext|>": 0,
            }.get(token, self.unk_token_id)

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            token_id = self.convert_tokens_to_ids(text)
            return [] if token_id == self.unk_token_id else [token_id]

    class _FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = _FakeTokenizer()

    class _FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    processor = _FakeProcessor()

    monkeypatch.setattr(
        infer_runtime,
        "Qwen3VLForConditionalGeneration",
        types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeModel()),
    )
    monkeypatch.setattr(
        infer_runtime,
        "AutoProcessor",
        types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
    )

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
            backend_type="hf",
            detect_samples=1,
        ),
        GenerationConfig(max_new_tokens=1),
    )

    engine.load_model()

    assert processor.tokenizer.padding_side == "left"
    assert processor.tokenizer.pad_token_id == 0
    assert processor.tokenizer.pad_token_id != processor.tokenizer.eos_token_id


def test_infer_distributed_merge_preserves_order_and_trace(tmp_path, monkeypatch):
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    for i in range(4):
        _write_img(tmp_path / f"img_{i}.png")

    gt_path = tmp_path / "gt.jsonl"
    with gt_path.open("w", encoding="utf-8") as f:
        for i in range(4):
            rec = {
                "images": [f"img_{i}.png"],
                "width": 32,
                "height": 32,
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": f"obj-{i}"}],
                "image_id": i,
                "metadata": {"sample_index": i},
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    out_path = tmp_path / "gt_vs_pred.jsonl"
    trace_path = tmp_path / "pred_token_trace.jsonl"
    summary_path = tmp_path / "summary.json"

    def _build_engine(rank: int):
        inf_cfg = InferenceConfig(
            gt_jsonl=str(gt_path),
            model_checkpoint="dummy",
            mode="text",
            pred_coord_mode="auto",
            out_path=str(out_path),
            pred_token_trace_path=str(trace_path),
            summary_path=str(summary_path),
            device="cpu",
            limit=3,
            backend_type="hf",
            backend={},
            detect_samples=1,
            rank=rank,
            local_rank=rank,
            world_size=2,
            distributed_enabled=True,
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
        monkeypatch.setattr(engine, "load_model", lambda: None)

        def _fake_generate_batch(images):
            text = '{"objects": [{"desc": "obj", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_10|>, <|coord_10|>]}]}<|im_end|>'
            return [
                GenerationResult(
                    text=text,
                    generated_token_text=[f"rank-{rank}", "tok"],
                    token_logprobs=[-0.1, -0.2],
                    error=None,
                )
                for _ in images
            ]

        monkeypatch.setattr(engine, "_generate_batch", _fake_generate_batch)
        return engine

    rank1_engine = _build_engine(rank=1)
    rank1_engine.infer()
    assert not out_path.exists()

    rank0_engine = _build_engine(rank=0)
    got_out, got_summary = rank0_engine.infer()

    assert got_out == out_path
    assert got_summary == summary_path

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["image"] for row in rows] == ["img_0.png", "img_1.png", "img_2.png"]
    assert [row["image_id"] for row in rows] == [0, 1, 2]
    assert all("metadata" in row for row in rows)
    assert all("_coordexp_source_index" not in row for row in rows)

    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [trace["line_idx"] for trace in traces] == [0, 1, 2]
    assert all("_coordexp_source_index" not in trace for trace in traces)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_emitted"] == 3
    assert summary["distributed"]["enabled"] is True
    assert summary["distributed"]["world_size"] == 2


def test_infer_distributed_tqdm_uses_global_progress_on_rank_zero(tmp_path, monkeypatch):
    monkeypatch.delenv("ROOT_IMAGE_DIR", raising=False)

    for i in range(4):
        _write_img(tmp_path / f"img_{i}.png")

    gt_path = tmp_path / "gt.jsonl"
    with gt_path.open("w", encoding="utf-8") as f:
        for i in range(4):
            rec = {
                "images": [f"img_{i}.png"],
                "width": 32,
                "height": 32,
                "objects": [{"bbox_2d": [0, 0, 10, 10], "desc": f"obj-{i}"}],
            }
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    out_path = tmp_path / "gt_vs_pred.jsonl"
    summary_path = tmp_path / "summary.json"

    tqdm_events: list[dict[str, object]] = []

    class _FakeTqdm:
        def __init__(self, *args, **kwargs):
            self.disable = bool(kwargs.get("disable", False))
            self.total = kwargs.get("total")
            self.updates: list[int] = []
            tqdm_events.append(
                {
                    "disable": self.disable,
                    "total": self.total,
                    "updates": self.updates,
                }
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            self.updates.append(int(n))

    monkeypatch.setattr(infer_runtime, "tqdm", _FakeTqdm)

    inf_cfg = InferenceConfig(
        gt_jsonl=str(gt_path),
        model_checkpoint="dummy",
        mode="text",
        pred_coord_mode="auto",
        out_path=str(out_path),
        summary_path=str(summary_path),
        device="cpu",
        limit=3,
        backend_type="hf",
        backend={},
        detect_samples=1,
        rank=0,
        local_rank=0,
        world_size=2,
        distributed_enabled=True,
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
    monkeypatch.setattr(engine, "load_model", lambda: None)
    monkeypatch.setattr(
        infer_runtime,
        "wait_for_offline_inference_distributed_manifests",
        lambda *, owner, base_out_path: [
            out_path.parent / "shards" / "rank_00000" / "manifest.json"
        ],
    )
    monkeypatch.setattr(
        infer_runtime,
        "merge_offline_inference_distributed_outputs",
        lambda **kwargs: make_offline_run_counters(),
    )

    def _fake_generate_batch(images):
        text = '{"objects": [{"desc": "obj", "bbox_2d": [<|coord_0|>, <|coord_0|>, <|coord_10|>, <|coord_10|>]}]}<|im_end|>'
        return [GenerationResult(text=text, error=None) for _ in images]

    monkeypatch.setattr(engine, "_generate_batch", _fake_generate_batch)

    engine.infer()

    assert len(tqdm_events) == 1
    assert tqdm_events[0]["disable"] is False
    assert tqdm_events[0]["total"] == 3
    assert sum(tqdm_events[0]["updates"]) == 3
