from __future__ import annotations

import torch

from src.trainers.metrics.mixins import _validate_batch_contract


class _DummyEmbeddings:
    def __init__(self, rows: int) -> None:
        self.weight = torch.zeros((rows, 8), dtype=torch.float32)


class _DummyTextConfig:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)


class _DummyConfig:
    def __init__(self, vocab_size: int) -> None:
        self.text_config = _DummyTextConfig(vocab_size)


class _DummyModel:
    def __init__(self, vocab_size: int) -> None:
        self.config = _DummyConfig(vocab_size)
        self._emb = _DummyEmbeddings(vocab_size)

    def get_input_embeddings(self):
        return self._emb


def test_validate_batch_contract_accepts_valid_causal_lm_batch() -> None:
    model = _DummyModel(vocab_size=32)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3, 4]], dtype=torch.long),
    }

    _validate_batch_contract(model=model, inputs=inputs)


def test_validate_batch_contract_rejects_label_id_outside_vocab() -> None:
    model = _DummyModel(vocab_size=32)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 32, 4]], dtype=torch.long),
    }

    try:
        _validate_batch_contract(model=model, inputs=inputs)
    except ValueError as exc:
        assert "target range" in str(exc)
        assert "bad_values=[32]" in str(exc)
    else:
        raise AssertionError("expected ValueError for out-of-vocab label")


class _DummyImageProcessor:
    def __init__(self, merge_size: int = 2) -> None:
        self.merge_size = int(merge_size)


class _DummyProcessor:
    def __init__(self, merge_size: int = 2) -> None:
        self.image_processor = _DummyImageProcessor(merge_size)


class _DummyTemplate:
    def __init__(self, image_token_id: int = 9, merge_size: int = 2) -> None:
        self.image_token_id = int(image_token_id)
        self.processor = _DummyProcessor(merge_size)


def test_validate_batch_contract_accepts_valid_packed_multimodal_batch() -> None:
    model = _DummyModel(vocab_size=64)
    template = _DummyTemplate(image_token_id=9, merge_size=2)
    inputs = {
        "input_ids": torch.tensor([[9, 9, 1, 9, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 1, -100, 2, 3]], dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 4], [1, 2, 2]], dtype=torch.long),
        "pixel_values": torch.zeros((12, 1536), dtype=torch.float32),
        "position_ids": torch.zeros((3, 1, 6), dtype=torch.long),
        "text_position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
        "cu_seq_lens_q": torch.tensor([0, 3, 6], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 3, 6], dtype=torch.int32),
        "max_length_q": 3,
        "max_length_k": 3,
        "pack_num_samples": torch.tensor([2], dtype=torch.long),
    }

    _validate_batch_contract(model=model, inputs=inputs, template=template)


def test_validate_batch_contract_rejects_image_token_count_mismatch() -> None:
    model = _DummyModel(vocab_size=64)
    template = _DummyTemplate(image_token_id=9, merge_size=2)
    inputs = {
        "input_ids": torch.tensor([[9, 1, 2, 9, 3, 4]], dtype=torch.long),
        "labels": torch.tensor([[-100, 1, 2, -100, 3, 4]], dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 4], [1, 2, 2]], dtype=torch.long),
        "pixel_values": torch.zeros((12, 1536), dtype=torch.float32),
    }

    try:
        _validate_batch_contract(model=model, inputs=inputs, template=template)
    except ValueError as exc:
        assert "image token count" in str(exc)
        assert "expected=3 actual=2" in str(exc)
    else:
        raise AssertionError("expected ValueError for image token/grid mismatch")


def test_validate_batch_contract_rejects_cu_seq_reset_mismatch() -> None:
    model = _DummyModel(vocab_size=64)
    template = _DummyTemplate(image_token_id=9, merge_size=2)
    inputs = {
        "input_ids": torch.tensor([[9, 9, 1, 9, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 1, -100, 2, 3]], dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 4], [1, 2, 2]], dtype=torch.long),
        "pixel_values": torch.zeros((12, 1536), dtype=torch.float32),
        "position_ids": torch.zeros((3, 1, 6), dtype=torch.long),
        "text_position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
        "cu_seq_lens_q": torch.tensor([0, 2, 6], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 2, 6], dtype=torch.int32),
        "max_length_q": 4,
        "max_length_k": 4,
        "pack_num_samples": torch.tensor([2], dtype=torch.long),
    }

    try:
        _validate_batch_contract(model=model, inputs=inputs, template=template)
    except ValueError as exc:
        assert "reset points mismatch cu_seq_lens_q" in str(exc) or "boundaries mismatch input_ids" in str(exc)
    else:
        raise AssertionError("expected ValueError for cu_seq/text_position mismatch")
