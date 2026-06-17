from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tools" / "expand_coord_vocab.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "expand_coord_vocab_script",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)


class _DummyTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        self.additional_special_tokens: list[str] = []
        self._base_tokens = {str(i): i for i in range(min(vocab_size, 10))}

    def add_special_tokens(
        self,
        special_tokens_dict: dict[str, list[str]],
        replace_additional_special_tokens: bool = False,
    ) -> int:
        assert replace_additional_special_tokens is False
        added = 0
        for token in special_tokens_dict["additional_special_tokens"]:
            if token in self.additional_special_tokens:
                continue
            self.additional_special_tokens.append(token)
            self._vocab_size += 1
            added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in self.additional_special_tokens:
            return self._vocab_size - len(self.additional_special_tokens) + self.additional_special_tokens.index(token)
        if token in self._base_tokens:
            return self._base_tokens[token]
        raise KeyError(token)

    def __len__(self) -> int:
        return self._vocab_size

    def save_pretrained(self, output_dir: Path) -> None:
        (output_dir / "tokenizer.saved").write_text("ok\n", encoding="utf-8")


class _EmbeddingHandle:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight


class _DummyModel:
    def __init__(self, base_weight: torch.Tensor) -> None:
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self._input = _EmbeddingHandle(base_weight.clone())
        self._output = _EmbeddingHandle(base_weight.clone())
        self.last_mean_resizing: bool | None = None
        self.mean_resized_weight: torch.Tensor | None = None

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> _EmbeddingHandle:
        del pad_to_multiple_of
        self.last_mean_resizing = mean_resizing
        old_weight = self._input.weight
        added = new_num_tokens - old_weight.shape[0]
        if added <= 0:
            return self._input

        if mean_resizing:
            new_rows = torch.randn(added, old_weight.shape[1], dtype=old_weight.dtype)
        else:
            new_rows = torch.zeros(added, old_weight.shape[1], dtype=old_weight.dtype)
        resized = torch.cat([old_weight, new_rows], dim=0)
        self.mean_resized_weight = resized.clone()
        self._input = _EmbeddingHandle(resized)
        self._output = _EmbeddingHandle(resized.clone())
        return self._input

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self._output = self._input

    def get_input_embeddings(self) -> _EmbeddingHandle:
        return self._input

    def get_output_embeddings(self) -> _EmbeddingHandle:
        return self._output

    def save_pretrained(self, output_dir: Path) -> None:
        torch.save(
            {
                "embed_tokens.weight": self._input.weight.clone(),
                "lm_head.weight": self._output.weight.clone(),
            },
            output_dir / "weights.pt",
        )


def test_expand_coord_vocab_initializes_added_rows_deterministically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    def _run_once(dst_dir: Path) -> torch.Tensor:
        captured: dict[str, object] = {}

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(src: Path, trust_remote_code: bool) -> _DummyTokenizer:
                assert src == src_dir
                assert trust_remote_code is True
                return _DummyTokenizer(vocab_size=base_weight.shape[0])

        class _FakeQwen3VL:
            @staticmethod
            def from_pretrained(src: Path, trust_remote_code: bool) -> _DummyModel:
                assert src == src_dir
                assert trust_remote_code is True
                model = _DummyModel(base_weight=base_weight)
                captured["model"] = model
                return model

        monkeypatch.setattr(_SCRIPT_MODULE, "AutoTokenizer", _FakeAutoTokenizer)
        monkeypatch.setattr(
            _SCRIPT_MODULE,
            "Qwen3VLForConditionalGeneration",
            _FakeQwen3VL,
        )
        monkeypatch.setattr(
            _SCRIPT_MODULE,
            "parse_args",
            lambda: SimpleNamespace(
                src=src_dir,
                dst=dst_dir,
                num_bins=2,
                no_wildcard=False,
                coord_init="mean_resize",
                coord_init_frequencies=8,
                coord_init_scale=0.02,
                coord_init_seed=_SCRIPT_MODULE.TRANSFORMERS_RESIZE_SEED,
            ),
        )

        _SCRIPT_MODULE.main()
        saved = torch.load(dst_dir / "weights.pt")
        return (
            saved["embed_tokens.weight"][base_weight.shape[0] :],
            captured["model"].last_mean_resizing,
        )

    first_added, first_mean_resizing = _run_once(tmp_path / "out-1")
    second_added, second_mean_resizing = _run_once(tmp_path / "out-2")

    assert first_mean_resizing is True
    assert second_mean_resizing is True
    assert torch.equal(first_added, second_added)


def test_natural_adjacent_coord_init_overwrites_numeric_rows_only_and_records_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_weight = torch.arange(640, dtype=torch.float32).reshape(10, 64)
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    def _run_once(dst_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        captured: dict[str, object] = {}

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(src: Path, trust_remote_code: bool) -> _DummyTokenizer:
                assert src == src_dir
                assert trust_remote_code is True
                return _DummyTokenizer(vocab_size=base_weight.shape[0])

        class _FakeQwen3VL:
            @staticmethod
            def from_pretrained(src: Path, trust_remote_code: bool) -> _DummyModel:
                assert src == src_dir
                assert trust_remote_code is True
                model = _DummyModel(base_weight=base_weight)
                captured["model"] = model
                return model

        monkeypatch.setattr(_SCRIPT_MODULE, "AutoTokenizer", _FakeAutoTokenizer)
        monkeypatch.setattr(
            _SCRIPT_MODULE,
            "Qwen3VLForConditionalGeneration",
            _FakeQwen3VL,
        )
        monkeypatch.setattr(
            _SCRIPT_MODULE,
            "parse_args",
            lambda: SimpleNamespace(
                src=src_dir,
                dst=dst_dir,
                num_bins=120,
                no_wildcard=False,
                coord_init="natural_adjacent",
                coord_init_frequencies=4,
                coord_init_scale=0.02,
                coord_init_seed=123,
            ),
        )

        torch.manual_seed(999)
        _SCRIPT_MODULE.main()
        model = captured["model"]
        saved = torch.load(dst_dir / "weights.pt")
        weights = saved["embed_tokens.weight"]
        wildcard_row = weights[base_weight.shape[0]]
        coord_rows = weights[base_weight.shape[0] + 1 :]
        mean_resized_rows = model.mean_resized_weight[base_weight.shape[0] :]
        return weights, wildcard_row, coord_rows, mean_resized_rows

    first_weights, first_wildcard, first_coord_rows, first_mean_resized = _run_once(
        tmp_path / "out-1"
    )
    second_weights, second_wildcard, second_coord_rows, second_mean_resized = _run_once(
        tmp_path / "out-2"
    )

    assert torch.equal(first_weights, second_weights)
    assert torch.equal(first_wildcard, first_mean_resized[0])
    assert torch.equal(first_wildcard, second_wildcard)
    assert not torch.equal(first_coord_rows, first_mean_resized[1:])
    assert torch.equal(first_coord_rows, second_coord_rows)

    adjacent_distances = torch.linalg.vector_norm(
        first_coord_rows[1:] - first_coord_rows[:-1],
        dim=1,
    )
    far_distances = torch.linalg.vector_norm(
        first_coord_rows[100:] - first_coord_rows[:-100],
        dim=1,
    )
    assert adjacent_distances.mean() < far_distances.mean()

    assert torch.equal(
        first_weights,
        torch.load(tmp_path / "out-1" / "weights.pt")["lm_head.weight"],
    )
    metadata = json.loads((tmp_path / "out-1" / "coord_init.json").read_text())
    assert metadata == {
        "schema_version": 1,
        "coord_init": "natural_adjacent",
        "num_numeric_coord_tokens": 121,
        "num_frequencies": 4,
        "scale": 0.02,
        "seed": 123,
    }


def test_mean_resize_coord_init_does_not_write_init_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(src: Path, trust_remote_code: bool) -> _DummyTokenizer:
            assert src == src_dir
            assert trust_remote_code is True
            return _DummyTokenizer(vocab_size=base_weight.shape[0])

    class _FakeQwen3VL:
        @staticmethod
        def from_pretrained(src: Path, trust_remote_code: bool) -> _DummyModel:
            assert src == src_dir
            assert trust_remote_code is True
            return _DummyModel(base_weight=base_weight)

    monkeypatch.setattr(_SCRIPT_MODULE, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(
        _SCRIPT_MODULE,
        "Qwen3VLForConditionalGeneration",
        _FakeQwen3VL,
    )
    monkeypatch.setattr(
        _SCRIPT_MODULE,
        "parse_args",
        lambda: SimpleNamespace(
            src=src_dir,
            dst=tmp_path / "out",
            num_bins=2,
            no_wildcard=False,
            coord_init="mean_resize",
            coord_init_frequencies=8,
            coord_init_scale=0.02,
            coord_init_seed=_SCRIPT_MODULE.TRANSFORMERS_RESIZE_SEED,
        ),
    )

    _SCRIPT_MODULE.main()

    assert not (tmp_path / "out" / "coord_init.json").exists()
