from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from src.inference.backend import BackendLaunch
from src.qwen.coordinate_codebook import (
    install_coordinate_codebook,
    load_coordinate_codebook,
    save_coordinate_codebook,
)
from src.qwen.tokens import (
    DEFAULT_COORDINATE_TOKENS,
    DEFAULT_WRAPPER_TOKENS,
    QwenTokenIdentity,
)


HIDDEN = 4
COORDINATE_IDS = tuple(range(4, 1004))
VOCAB = 1004


class _Visual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.merger = nn.Linear(HIDDEN, HIDDEN, bias=False)


class _UntiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.model = nn.Module()
        self.model.add_module("visual", _Visual())

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, output_embeddings: nn.Module) -> None:
        self.lm_head = output_embeddings


def _identity() -> QwenTokenIdentity:
    return QwenTokenIdentity(
        required_tokens=(*DEFAULT_WRAPPER_TOKENS, *DEFAULT_COORDINATE_TOKENS),
        wrapper_token_ids={token: index for index, token in enumerate(DEFAULT_WRAPPER_TOKENS)},
        coordinate_token_ids=COORDINATE_IDS,
        im_end_newline_text="<|im_end|>\n",
        im_end_token_ids=(1004,),
        newline_token_ids=(1005,),
        im_end_newline_token_ids=(1004, 1005),
        tokenizer_vocab_size=VOCAB,
    )


def _launch(
    codebook_path: Path | None,
    embedding_path: Path,
    *,
    adapter_path: Path | None = None,
) -> BackendLaunch:
    hf_options = {
        "attn_implementation": "eager",
        "patch_embed_linearization": "disabled",
    }
    if codebook_path is not None:
        hf_options["coordinate_codebook_path"] = str(codebook_path)
    return BackendLaunch(
        backend="hf",
        model_path="/tmp/base",
        model_dtype="fp32",
        batch_size=1,
        generation_config_fingerprint="test",
        backend_options={
            "hf": hf_options,
        },
        adapter=(
            {"path": str(adapter_path), "name": "default"}
            if adapter_path is not None else None
        ),
        embedding_delta={"path": str(embedding_path)},
    )


def _write_embedding_metadata(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "special_token_embeddings.json").write_text(
        json.dumps({"tie_word_embeddings": False}), encoding="utf-8"
    )


def _write_codebook_payload(path: Path) -> None:
    source = _UntiedModel()
    from src.qwen.untied_embeddings import (
        SpecialTokenSelection,
        install_special_token_embedding_deltas,
    )

    selection = SpecialTokenSelection(
        token_strings=(*DEFAULT_WRAPPER_TOKENS, *DEFAULT_COORDINATE_TOKENS),
        token_ids=tuple(range(VOCAB)),
    )
    install_special_token_embedding_deltas(source, selection, tie_word_embeddings=False)
    install_coordinate_codebook(source, COORDINATE_IDS)
    save_coordinate_codebook(source, path)


def _load_with_real_wrappers(monkeypatch: pytest.MonkeyPatch, launch: BackendLaunch):
    import src.inference.hf_backend as backend
    import src.qwen.untied_embeddings as untied

    identity = _identity()

    def make_components(_options: object) -> SimpleNamespace:
        return SimpleNamespace(
            model=_UntiedModel(),
            token_identity=identity,
            base_model_path=Path("/tmp/base"),
        )

    def load_delta(*, qwen: SimpleNamespace, **_: object) -> dict[str, str]:
        from src.qwen.untied_embeddings import (
            SpecialTokenSelection,
            install_special_token_embedding_deltas,
        )

        selection = SpecialTokenSelection(
            token_strings=(*DEFAULT_WRAPPER_TOKENS, *DEFAULT_COORDINATE_TOKENS),
            token_ids=tuple(range(VOCAB)),
        )
        install_special_token_embedding_deltas(
            qwen.model, selection, tie_word_embeddings=False
        )
        return {"status": "loaded"}

    monkeypatch.setattr(backend, "load_qwen_components_from_options", make_components)
    monkeypatch.setattr(untied, "load_inference_embedding_delta", load_delta)
    monkeypatch.setattr(
        backend,
        "attach_dora_adapter",
        lambda *args, **kwargs: {"status": "adapter-loaded"},
    )
    return backend._load_hf_components(launch)


def test_hf_reload_installs_real_untied_wrappers_then_live_codebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codebook_path = tmp_path / "codebook"
    embedding_path = tmp_path / "embedding"
    _write_codebook_payload(codebook_path)
    _write_embedding_metadata(embedding_path)

    loaded = _load_with_real_wrappers(
        monkeypatch, _launch(codebook_path, embedding_path)
    )
    model = loaded.qwen.model
    assert model.coordinate_codebook is not None
    assert model.get_input_embeddings().shared_embed_delta is not model.get_output_embeddings().shared_embed_delta
    result = model.coordinate_codebook(
        torch.ones((1, HIDDEN)), torch.tensor([[1, 2, 2]])
    )
    assert result.shape == (1, HIDDEN)


def test_hf_reload_rejects_missing_or_stale_codebook_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedding_path = tmp_path / "embedding"
    _write_embedding_metadata(embedding_path)
    missing = tmp_path / "missing-codebook"
    with pytest.raises(FileNotFoundError):
        _load_with_real_wrappers(monkeypatch, _launch(missing, embedding_path))

    stale = tmp_path / "stale-codebook"
    _write_codebook_payload(stale)
    metadata_path = stale / "coordinate_codebook.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["hidden_size"] = HIDDEN + 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture metadata mismatch"):
        _load_with_real_wrappers(monkeypatch, _launch(stale, embedding_path))


def test_hf_reload_composition_marker_autoloads_sibling_and_requires_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    adapter_path = checkpoint / "adapter"
    embedding_path = checkpoint / "special_token_embeddings"
    codebook_path = checkpoint / "coordinate_codebook"
    adapter_path.mkdir(parents=True)
    _write_embedding_metadata(embedding_path)
    _write_codebook_payload(codebook_path)
    (checkpoint / "model_composition.json").write_text(
        json.dumps({
            "schema": 1,
            "coordinate_codebook": "coordinate_codebook",
            "untied_selected_rows": True,
        }),
        encoding="utf-8",
    )

    loaded = _load_with_real_wrappers(
        monkeypatch,
        _launch(None, embedding_path, adapter_path=adapter_path),
    )
    assert loaded.qwen.model.coordinate_codebook is not None

    # Removing the composed sibling must fail closed when no explicit override
    # is supplied; the marker is the source of truth for the required payload.
    for child in codebook_path.iterdir():
        child.unlink()
    codebook_path.rmdir()
    with pytest.raises(FileNotFoundError):
        _load_with_real_wrappers(
            monkeypatch,
            _launch(None, embedding_path, adapter_path=adapter_path),
        )


def test_codebook_reload_rejects_nonfinite_raw_gain(tmp_path: Path) -> None:
    payload = tmp_path / "codebook"
    model = _UntiedModel()
    from src.qwen.untied_embeddings import (
        SpecialTokenSelection,
        install_special_token_embedding_deltas,
    )

    selection = SpecialTokenSelection(
        token_strings=(*DEFAULT_WRAPPER_TOKENS, *DEFAULT_COORDINATE_TOKENS),
        token_ids=tuple(range(VOCAB)),
    )
    install_special_token_embedding_deltas(model, selection, tie_word_embeddings=False)
    install_coordinate_codebook(model, COORDINATE_IDS)
    save_coordinate_codebook(model, payload)
    save_file({"raw_gain": torch.tensor(float("nan"))}, str(payload / "coordinate_codebook.safetensors"))
    with pytest.raises(ValueError, match="raw_gain must be finite"):
        load_coordinate_codebook(model, payload)
