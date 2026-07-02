"""Selected special-token embedding deltas for Qwen3-VL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from src.common.errors import RuntimeContractError
from src.config.models import SpecialTokenEmbeddingsConfig
from src.config.models import SpecialTokenEmbeddingGroupsConfig
from src.qwen.tokens import (
    DEFAULT_COORDINATE_TOKENS,
    DEFAULT_WRAPPER_TOKENS,
    QwenTokenIdentity,
)


SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS = "special_token_embeddings.safetensors"
SPECIAL_TOKEN_EMBEDDINGS_JSON = "special_token_embeddings.json"
DEFAULT_EMBED_DELTA_TENSOR_KEY = "shared_embed_delta"
SPECIAL_TOKEN_EMBEDDING_SEMANTICS = "additive_delta"
DEFAULT_SPECIAL_TOKEN_EMBEDDING_SOURCE_STUDY_PATH = Path(
    "docs/architecture/proposals/2026-06-27-coordexp-swift/source-studies/"
    "special-token-embeddings.md"
)
DEFAULT_SPECIAL_TOKEN_EMBEDDING_PROBE_RECEIPT_PATH = Path(
    "outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json"
)


@dataclass(frozen=True)
class SpecialTokenSelection:
    token_strings: tuple[str, ...]
    token_ids: tuple[int, ...]

    def __init__(
        self,
        *,
        token_strings: Sequence[str],
        token_ids: Sequence[int],
    ) -> None:
        object.__setattr__(self, "token_strings", tuple(str(item) for item in token_strings))
        object.__setattr__(self, "token_ids", tuple(int(item) for item in token_ids))
        self._validate()

    def __len__(self) -> int:
        return len(self.token_ids)

    def to_artifact_dict(self) -> dict[str, Any]:
        coord_ids = self.token_ids[len(DEFAULT_WRAPPER_TOKENS) :]
        coord_contiguous = bool(coord_ids) and coord_ids == tuple(
            range(coord_ids[0], coord_ids[-1] + 1)
        )
        return {
            "selected_token_count": len(self),
            "token_strings": list(self.token_strings),
            "token_ids": list(self.token_ids),
            "wrapper_token_ids": {
                token: token_id
                for token, token_id in zip(
                    self.token_strings[: len(DEFAULT_WRAPPER_TOKENS)],
                    self.token_ids[: len(DEFAULT_WRAPPER_TOKENS)],
                    strict=True,
                )
            },
            "coord_token_count": len(coord_ids),
            "coord_token_id_min": None if not coord_ids else coord_ids[0],
            "coord_token_id_max": None if not coord_ids else coord_ids[-1],
            "coord_token_ids_contiguous": coord_contiguous,
        }

    def _validate(self) -> None:
        if len(self.token_strings) != len(self.token_ids):
            raise RuntimeContractError(
                "special-token selection strings and ids must have the same length",
                code="special_token_embeddings.selection_shape",
                context={
                    "token_string_count": len(self.token_strings),
                    "token_id_count": len(self.token_ids),
                },
            )
        if not self.token_ids:
            raise RuntimeContractError(
                "special-token selection must not be empty",
                code="special_token_embeddings.selection_empty",
            )
        if len(set(self.token_strings)) != len(self.token_strings):
            raise RuntimeContractError(
                "special-token selection strings must be unique",
                code="special_token_embeddings.duplicate_token_string",
            )
        if len(set(self.token_ids)) != len(self.token_ids):
            raise RuntimeContractError(
                "special-token selection ids must be unique",
                code="special_token_embeddings.duplicate_token_id",
            )
        if any(token_id < 0 for token_id in self.token_ids):
            raise RuntimeContractError(
                "special-token selection ids must be non-negative",
                code="special_token_embeddings.negative_token_id",
                context={"token_ids": list(self.token_ids)},
            )


@dataclass(frozen=True)
class SpecialTokenEmbeddingSourceGateEvidence:
    source_study_passed: bool
    roundtrip_probe_passed: bool
    probe_receipt: Mapping[str, Any] | None = None


def load_default_special_token_embedding_source_gate_evidence(
    repo_root: str | Path,
) -> SpecialTokenEmbeddingSourceGateEvidence:
    root = Path(repo_root).expanduser().resolve()
    source_study_path = root / DEFAULT_SPECIAL_TOKEN_EMBEDDING_SOURCE_STUDY_PATH
    probe_receipt_path = root / DEFAULT_SPECIAL_TOKEN_EMBEDDING_PROBE_RECEIPT_PATH
    source_study_passed = _special_token_embedding_source_study_is_passed(
        source_study_path
    )
    probe_receipt = (
        _load_probe_receipt(probe_receipt_path)
        if probe_receipt_path.exists()
        else None
    )
    return SpecialTokenEmbeddingSourceGateEvidence(
        source_study_passed=source_study_passed,
        roundtrip_probe_passed=_special_token_embedding_probe_is_passed(probe_receipt),
        probe_receipt=probe_receipt,
    )


@dataclass(frozen=True)
class SpecialTokenEmbeddingInstallReceipt:
    semantics: str
    tensor_key: str
    tie_word_embeddings: bool
    token_selection: SpecialTokenSelection
    delta_shape: tuple[int, int]
    delta_dtype: str
    delta_parameter_names: tuple[str, ...]
    base_embedding_parameter_name: str | None
    base_lm_head_parameter_name: str | None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "semantics": self.semantics,
            "tensor_key": self.tensor_key,
            "tie_word_embeddings": self.tie_word_embeddings,
            "token_selection": self.token_selection.to_artifact_dict(),
            "delta_shape": list(self.delta_shape),
            "delta_dtype": self.delta_dtype,
            "delta_parameter_names": list(self.delta_parameter_names),
            "base_embedding_parameter_name": self.base_embedding_parameter_name,
            "base_lm_head_parameter_name": self.base_lm_head_parameter_name,
        }

    def to_metadata_dict(
        self,
        *,
        base_model_path: Path | str | None,
        base_config_sha256: str | None,
        tokenizer_sha256: str | None,
    ) -> dict[str, Any]:
        return {
            "semantics": self.semantics,
            "tensor_key": self.tensor_key,
            "tensor_shape": list(self.delta_shape),
            "tensor_dtype": self.delta_dtype,
            "token_strings": list(self.token_selection.token_strings),
            "token_ids": list(self.token_selection.token_ids),
            "base_model_path": None if base_model_path is None else str(base_model_path),
            "base_config_sha256": base_config_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


@dataclass(frozen=True)
class SpecialTokenEmbeddingInstallResult:
    model: nn.Module
    input_wrapper: "SelectedDeltaInputEmbedding"
    output_wrapper: "SelectedDeltaOutputHead"
    shared_embed_delta: nn.Parameter
    receipt: SpecialTokenEmbeddingInstallReceipt


@dataclass(frozen=True)
class SpecialTokenEmbeddingPayloadReceipt:
    tensor_path: Path
    metadata_path: Path
    tensor_key: str
    tensor_shape: tuple[int, int]
    tensor_dtype: str
    metadata: Mapping[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "tensor_path": str(self.tensor_path),
            "metadata_path": str(self.metadata_path),
            "tensor_key": self.tensor_key,
            "tensor_shape": list(self.tensor_shape),
            "tensor_dtype": self.tensor_dtype,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SpecialTokenEmbeddingLoadReceipt:
    loaded: bool
    tensor_path: Path
    metadata_path: Path
    tensor_key: str
    tensor_shape: tuple[int, int]
    tensor_dtype: str

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "tensor_path": str(self.tensor_path),
            "metadata_path": str(self.metadata_path),
            "tensor_key": self.tensor_key,
            "tensor_shape": list(self.tensor_shape),
            "tensor_dtype": self.tensor_dtype,
        }


@dataclass(frozen=True)
class InferenceEmbeddingDeltaIdentityReceipt:
    status: str
    delta_path: Path
    metadata_path: Path
    metadata: Mapping[str, Any]
    base_model_path: str | None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "delta_path": str(self.delta_path),
            "metadata_path": str(self.metadata_path),
            "metadata": dict(self.metadata),
            "base_model_path": self.base_model_path,
        }


def validate_inference_embedding_delta_identity(
    *,
    config: Any,
    qwen: Any,
) -> dict[str, Any]:
    embedding_delta = getattr(config, "embedding_delta", None)
    if embedding_delta is None:
        raise RuntimeContractError(
            "inference embedding-delta identity validation requires delta config",
            code="special_token_embeddings.inference_config_missing",
        )
    delta_path = Path(embedding_delta.path)
    metadata_path = _inference_delta_metadata_path(delta_path)
    metadata = _load_metadata(metadata_path)
    _validate_inference_delta_metadata(metadata, qwen=qwen)
    receipt = InferenceEmbeddingDeltaIdentityReceipt(
        status="validated",
        delta_path=delta_path,
        metadata_path=metadata_path,
        metadata=metadata,
        base_model_path=_qwen_base_model_path(qwen),
    )
    return receipt.to_artifact_dict()


class SelectedDeltaInputEmbedding(nn.Module):
    def __init__(
        self,
        base: nn.Embedding,
        selection: SpecialTokenSelection,
        delta: nn.Parameter,
    ) -> None:
        super().__init__()
        self.base = base
        self.selection = selection
        self.register_buffer(
            "token_to_row",
            _build_token_to_row(selection.token_ids),
            persistent=False,
        )
        self.shared_embed_delta = delta

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_out = self.base(input_ids)
        token_to_row = self.token_to_row.to(input_ids.device)
        in_range = (input_ids >= 0) & (input_ids < token_to_row.numel())
        safe_ids = input_ids.clamp_min(0).clamp_max(token_to_row.numel() - 1)
        rows = token_to_row[safe_ids]
        selected = in_range & (rows >= 0)
        safe_rows = rows.clamp_min(0)
        delta_out = self.shared_embed_delta.to(base_out.dtype)[safe_rows]
        return base_out + delta_out * selected.unsqueeze(-1).to(base_out.dtype)


class SelectedDeltaOutputHead(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        selection: SpecialTokenSelection,
        delta: nn.Parameter,
    ) -> None:
        super().__init__()
        self.base = base
        self.selection = selection
        self.register_buffer(
            "selected_token_ids",
            torch.tensor(selection.token_ids, dtype=torch.long),
            persistent=False,
        )
        self.shared_embed_delta = delta

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.base(hidden_states)
        correction = hidden_states.to(self.shared_embed_delta.dtype) @ (
            self.shared_embed_delta.t()
        )
        correction = correction.to(logits.dtype)
        selected_ids = self.selected_token_ids.to(logits.device)
        scatter_index = selected_ids.view(*([1] * (correction.ndim - 1)), -1)
        scatter_index = scatter_index.expand_as(correction)
        logits.scatter_add_(-1, scatter_index, correction)
        return logits


def build_default_special_token_selection(
    config: SpecialTokenEmbeddingsConfig,
    token_identity: QwenTokenIdentity,
) -> SpecialTokenSelection:
    if config.groups.wrapper_tokens != "default_object_box_wrappers":
        raise RuntimeContractError(
            "unsupported special-token wrapper group",
            code="special_token_embeddings.wrapper_group_unsupported",
            context={"wrapper_tokens": config.groups.wrapper_tokens},
        )
    if config.groups.coordinate_tokens != "default_coord_0_999":
        raise RuntimeContractError(
            "unsupported coordinate-token group",
            code="special_token_embeddings.coord_group_unsupported",
            context={"coordinate_tokens": config.groups.coordinate_tokens},
        )
    return SpecialTokenSelection(
        token_strings=(*DEFAULT_WRAPPER_TOKENS, *DEFAULT_COORDINATE_TOKENS),
        token_ids=(
            tuple(token_identity.wrapper_token_ids[token] for token in DEFAULT_WRAPPER_TOKENS)
            + token_identity.coordinate_token_ids
        ),
    )


def install_special_token_embedding_deltas(
    model: nn.Module,
    selection: SpecialTokenSelection,
    *,
    source_gate: SpecialTokenEmbeddingSourceGateEvidence,
) -> SpecialTokenEmbeddingInstallResult:
    _validate_source_gate(source_gate, selection)
    preexisting_trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    base_embedding = _input_embedding(model)
    output_head = _output_head(model)
    _validate_tied_base_weights(base_embedding, output_head)
    _validate_selection_in_vocab(selection, base_embedding, output_head)

    base_embedding.weight.requires_grad_(False)
    output_head.weight.requires_grad_(False)
    if output_head.bias is not None:
        output_head.bias.requires_grad_(False)

    delta = nn.Parameter(
        torch.zeros(
            len(selection),
            int(base_embedding.embedding_dim),
            device=base_embedding.weight.device,
            dtype=base_embedding.weight.dtype,
        )
    )
    input_wrapper = SelectedDeltaInputEmbedding(base_embedding, selection, delta)
    output_wrapper = SelectedDeltaOutputHead(output_head, selection, delta)
    model.set_input_embeddings(input_wrapper)
    model.set_output_embeddings(output_wrapper)

    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    expected_name = _delta_parameter_name(model, delta)
    unexpected_trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter is not delta
        and id(parameter) not in preexisting_trainable_ids
    ]
    if expected_name not in trainable_names or unexpected_trainable_names:
        raise RuntimeContractError(
            "special-token embedding setup produced an unexpected trainable surface",
            code="special_token_embeddings.trainable_surface",
            context={
                "trainable_names": list(trainable_names),
                "expected_delta_parameter_name": expected_name,
                "unexpected_trainable_names": unexpected_trainable_names,
            },
        )

    receipt = SpecialTokenEmbeddingInstallReceipt(
        semantics=SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
        tensor_key=DEFAULT_EMBED_DELTA_TENSOR_KEY,
        tie_word_embeddings=True,
        token_selection=selection,
        delta_shape=(len(selection), int(base_embedding.embedding_dim)),
        delta_dtype=_dtype_name(delta.dtype),
        delta_parameter_names=(expected_name,),
        base_embedding_parameter_name=_parameter_name(model, base_embedding.weight),
        base_lm_head_parameter_name=_parameter_name(model, output_head.weight),
    )
    return SpecialTokenEmbeddingInstallResult(
        model=model,
        input_wrapper=input_wrapper,
        output_wrapper=output_wrapper,
        shared_embed_delta=delta,
        receipt=receipt,
    )


def save_special_token_embedding_deltas(
    result: SpecialTokenEmbeddingInstallResult,
    output_dir: Path,
    *,
    base_model_path: Path | str | None,
    base_config_sha256: str | None = None,
    tokenizer_sha256: str | None = None,
) -> SpecialTokenEmbeddingPayloadReceipt:
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS
    metadata_path = output_dir / SPECIAL_TOKEN_EMBEDDINGS_JSON
    save_file(
        {DEFAULT_EMBED_DELTA_TENSOR_KEY: result.shared_embed_delta.detach().cpu()},
        str(tensor_path),
    )
    metadata = result.receipt.to_metadata_dict(
        base_model_path=base_model_path,
        base_config_sha256=base_config_sha256,
        tokenizer_sha256=tokenizer_sha256,
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SpecialTokenEmbeddingPayloadReceipt(
        tensor_path=tensor_path,
        metadata_path=metadata_path,
        tensor_key=DEFAULT_EMBED_DELTA_TENSOR_KEY,
        tensor_shape=tuple(int(item) for item in result.shared_embed_delta.shape),
        tensor_dtype=_dtype_name(result.shared_embed_delta.dtype),
        metadata=metadata,
    )


def load_special_token_embedding_deltas(
    result: SpecialTokenEmbeddingInstallResult,
    payload_dir: Path,
    *,
    expected_base_model_path: Path | str | None = None,
    expected_base_config_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
) -> SpecialTokenEmbeddingLoadReceipt:
    tensor_path = payload_dir / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS
    metadata_path = payload_dir / SPECIAL_TOKEN_EMBEDDINGS_JSON
    metadata = _load_metadata(metadata_path)
    _validate_metadata(
        metadata,
        result=result,
        expected_base_model_path=expected_base_model_path,
        expected_base_config_sha256=expected_base_config_sha256,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
    )
    tensors = load_file(str(tensor_path), device=str(result.shared_embed_delta.device))
    allowed_keys = {DEFAULT_EMBED_DELTA_TENSOR_KEY}
    if set(tensors) != allowed_keys:
        raise RuntimeContractError(
            "special-token embedding payload must contain only compact delta tensors",
            code="special_token_embeddings.unexpected_tensor_keys",
            context={
                "tensor_path": str(tensor_path),
                "actual_keys": sorted(tensors),
                "expected_keys": sorted(allowed_keys),
            },
        )
    loaded_delta = tensors[DEFAULT_EMBED_DELTA_TENSOR_KEY]
    if _dtype_name(loaded_delta.dtype) != metadata.get("tensor_dtype"):
        raise RuntimeContractError(
            "special-token embedding tensor dtype does not match metadata",
            code="special_token_embeddings.dtype_mismatch",
            context={
                "metadata_dtype": metadata.get("tensor_dtype"),
                "tensor_dtype": _dtype_name(loaded_delta.dtype),
            },
        )
    expected_shape = tuple(int(item) for item in result.shared_embed_delta.shape)
    if tuple(int(item) for item in loaded_delta.shape) != expected_shape:
        raise RuntimeContractError(
            "special-token embedding delta tensor shape mismatch",
            code="special_token_embeddings.tensor_shape_mismatch",
            context={
                "expected_shape": list(expected_shape),
                "actual_shape": [int(item) for item in loaded_delta.shape],
            },
        )
    with torch.no_grad():
        result.shared_embed_delta.copy_(
            loaded_delta.to(
                device=result.shared_embed_delta.device,
                dtype=result.shared_embed_delta.dtype,
            )
        )
    return SpecialTokenEmbeddingLoadReceipt(
        loaded=True,
        tensor_path=tensor_path,
        metadata_path=metadata_path,
        tensor_key=DEFAULT_EMBED_DELTA_TENSOR_KEY,
        tensor_shape=expected_shape,
        tensor_dtype=_dtype_name(result.shared_embed_delta.dtype),
    )


def _validate_source_gate(
    evidence: SpecialTokenEmbeddingSourceGateEvidence,
    selection: SpecialTokenSelection,
) -> None:
    missing: list[str] = []
    if not evidence.source_study_passed:
        missing.append("source_study")
    if not evidence.roundtrip_probe_passed:
        missing.append("roundtrip_probe")
    receipt = dict(evidence.probe_receipt or {})
    if not receipt:
        missing.append("probe_receipt")
    elif receipt.get("ok") is not True:
        missing.append("probe_receipt_ok")
    if missing:
        raise RuntimeContractError(
            "special-token embedding source gate has not passed",
            code="special_token_embeddings.source_gate_missing",
            context={"missing": missing},
        )
    expected_semantics = receipt.get("semantics")
    if expected_semantics != SPECIAL_TOKEN_EMBEDDING_SEMANTICS:
        raise RuntimeContractError(
            "special-token embedding source gate records unsupported semantics",
            code="special_token_embeddings.source_gate_semantics",
            context={
                "expected_semantics": SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
                "actual_semantics": expected_semantics,
            },
        )
    selected_count = receipt.get("num_selected_tokens")
    if selected_count is not None and int(selected_count) != len(selection):
        raise RuntimeContractError(
            "special-token embedding source gate selected-token count mismatch",
            code="special_token_embeddings.source_gate_selected_count",
            context={
                "expected_selected_count": len(selection),
                "actual_selected_count": selected_count,
            },
        )


def _special_token_embedding_source_study_is_passed(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    required_phrases = (
        "custom Qwen wrapper pair as V1 recommendation",
        "semantics: additive_delta",
        "outputs/probes/coordexp_swift/special_token_embeddings_roundtrip/receipt.json",
    )
    return all(phrase in text for phrase in required_phrases)


def _load_probe_receipt(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeContractError(
            "special-token embedding probe receipt must be a JSON object",
            code="special_token_embeddings.probe_receipt_shape",
            context={"path": str(path), "value_type": type(payload).__name__},
        )
    return payload


def _special_token_embedding_probe_is_passed(
    probe_receipt: Mapping[str, Any] | None,
) -> bool:
    if probe_receipt is None:
        return False
    payload = probe_receipt.get("payload")
    return (
        probe_receipt.get("ok") is True
        and probe_receipt.get("semantics") == SPECIAL_TOKEN_EMBEDDING_SEMANTICS
        and probe_receipt.get("num_selected_tokens") == 1004
        and probe_receipt.get("runtime_tied_input_lm_head_identity") is True
        and isinstance(payload, Mapping)
        and payload.get("safetensors") is not None
        and payload.get("metadata") is not None
    )


def _input_embedding(model: nn.Module) -> nn.Embedding:
    embedding = model.get_input_embeddings()
    if not isinstance(embedding, nn.Embedding):
        raise RuntimeContractError(
            "special-token embedding deltas require a direct nn.Embedding input module",
            code="special_token_embeddings.input_embedding_type",
            context={"module_class": type(embedding).__name__},
        )
    return embedding


def _output_head(model: nn.Module) -> nn.Linear:
    output = model.get_output_embeddings()
    if not isinstance(output, nn.Linear):
        raise RuntimeContractError(
            "special-token embedding deltas require a direct nn.Linear output head",
            code="special_token_embeddings.output_head_type",
            context={"module_class": type(output).__name__},
        )
    return output


def _validate_tied_base_weights(embedding: nn.Embedding, output_head: nn.Linear) -> None:
    if embedding.weight is not output_head.weight:
        raise RuntimeContractError(
            "V1 special-token embedding deltas require tied input embedding and lm_head weights",
            code="special_token_embeddings.untied_unsupported",
            context={
                "input_weight_shape": [int(item) for item in embedding.weight.shape],
                "output_weight_shape": [int(item) for item in output_head.weight.shape],
            },
        )


def _validate_selection_in_vocab(
    selection: SpecialTokenSelection,
    embedding: nn.Embedding,
    output_head: nn.Linear,
) -> None:
    max_selected_id = max(selection.token_ids)
    embedding_vocab_size = int(embedding.num_embeddings)
    output_vocab_size = int(output_head.out_features)
    if max_selected_id >= embedding_vocab_size or max_selected_id >= output_vocab_size:
        raise RuntimeContractError(
            "selected special-token id is outside embedding or output-head vocabulary",
            code="special_token_embeddings.token_id_out_of_range",
            context={
                "max_selected_token_id": max_selected_id,
                "embedding_vocab_size": embedding_vocab_size,
                "output_vocab_size": output_vocab_size,
            },
        )


def _build_token_to_row(selected_token_ids: tuple[int, ...]) -> torch.Tensor:
    mapping = torch.full((max(selected_token_ids) + 1,), -1, dtype=torch.long)
    mapping[torch.tensor(selected_token_ids, dtype=torch.long)] = torch.arange(
        len(selected_token_ids),
        dtype=torch.long,
    )
    return mapping


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeContractError(
            "special-token embedding metadata file is missing",
            code="special_token_embeddings.metadata_missing",
            context={"metadata_path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeContractError(
            "special-token embedding metadata must be a JSON object",
            code="special_token_embeddings.metadata_shape",
            context={"metadata_path": str(path)},
        )
    return payload


def _inference_delta_metadata_path(delta_path: Path) -> Path:
    if delta_path.is_dir():
        return delta_path / SPECIAL_TOKEN_EMBEDDINGS_JSON
    if delta_path.name == SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS:
        return delta_path.with_name(SPECIAL_TOKEN_EMBEDDINGS_JSON)
    return delta_path / SPECIAL_TOKEN_EMBEDDINGS_JSON


def _validate_inference_delta_metadata(metadata: Mapping[str, Any], *, qwen: Any) -> None:
    required = (
        "semantics",
        "tensor_key",
        "tensor_shape",
        "tensor_dtype",
        "token_strings",
        "token_ids",
        "base_model_path",
        "base_config_sha256",
        "tokenizer_sha256",
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise RuntimeContractError(
            "special-token embedding metadata is missing required inference identity fields",
            code="special_token_embeddings.inference_identity_missing",
            context={"missing_fields": missing},
        )
    if metadata.get("semantics") != SPECIAL_TOKEN_EMBEDDING_SEMANTICS:
        raise RuntimeContractError(
            "special-token embedding metadata records unsupported semantics",
            code="special_token_embeddings.inference_semantics_mismatch",
            context={
                "expected": SPECIAL_TOKEN_EMBEDDING_SEMANTICS,
                "actual": metadata.get("semantics"),
            },
        )
    base_model_path = _qwen_base_model_path(qwen)
    if base_model_path is not None and metadata.get("base_model_path") != base_model_path:
        raise RuntimeContractError(
            "special-token embedding metadata base model does not match runtime base",
            code="special_token_embeddings.identity_mismatch",
            context={
                "field": "base_model_path",
                "expected": base_model_path,
                "actual": metadata.get("base_model_path"),
            },
        )
    _validate_runtime_sha_field(
        metadata,
        qwen=qwen,
        field="base_config_sha256",
    )
    _validate_runtime_sha_field(
        metadata,
        qwen=qwen,
        field="tokenizer_sha256",
    )
    token_identity = _qwen_token_identity(qwen)
    if token_identity is not None:
        expected_selection = build_default_special_token_selection(
            SpecialTokenEmbeddingsConfig(
                groups=SpecialTokenEmbeddingGroupsConfig(
                    coordinate_tokens="default_coord_0_999",
                    wrapper_tokens="default_object_box_wrappers",
                )
            ),
            token_identity,
        )
        if metadata.get("token_strings") != list(expected_selection.token_strings):
            raise RuntimeContractError(
                "special-token embedding metadata token strings do not match runtime tokenizer",
                code="special_token_embeddings.identity_mismatch",
                context={"field": "token_strings"},
            )
        if metadata.get("token_ids") != list(expected_selection.token_ids):
            raise RuntimeContractError(
                "special-token embedding metadata token ids do not match runtime tokenizer",
                code="special_token_embeddings.identity_mismatch",
                context={"field": "token_ids"},
            )


def _validate_runtime_sha_field(
    metadata: Mapping[str, Any],
    *,
    qwen: Any,
    field: str,
) -> None:
    expected = _qwen_identity_field(qwen, field)
    if expected is None:
        raise RuntimeContractError(
            "runtime Qwen identity is missing required SHA evidence for embedding delta",
            code="special_token_embeddings.runtime_identity_missing",
            context={"field": field},
        )
    actual = metadata.get(field)
    if actual != expected:
        raise RuntimeContractError(
            "special-token embedding metadata SHA identity does not match runtime Qwen identity",
            code="special_token_embeddings.identity_mismatch",
            context={
                "field": field,
                "expected": expected,
                "actual": actual,
            },
        )


def _qwen_identity_field(qwen: Any, field: str) -> str | None:
    if isinstance(qwen, Mapping):
        value = qwen.get(field)
    else:
        value = getattr(qwen, field, None)
    return None if value is None else str(value)


def _qwen_base_model_path(qwen: Any) -> str | None:
    if isinstance(qwen, Mapping):
        value = qwen.get("base_model_path")
    else:
        value = getattr(qwen, "base_model_path", None)
    return None if value is None else str(value)


def _qwen_token_identity(qwen: Any) -> QwenTokenIdentity | None:
    if isinstance(qwen, Mapping):
        value = qwen.get("token_identity")
    else:
        value = getattr(qwen, "token_identity", None)
    return value if isinstance(value, QwenTokenIdentity) else None


def _validate_metadata(
    metadata: Mapping[str, Any],
    *,
    result: SpecialTokenEmbeddingInstallResult,
    expected_base_model_path: Path | str | None,
    expected_base_config_sha256: str | None,
    expected_tokenizer_sha256: str | None,
) -> None:
    expected = result.receipt.to_metadata_dict(
        base_model_path=expected_base_model_path,
        base_config_sha256=expected_base_config_sha256,
        tokenizer_sha256=expected_tokenizer_sha256,
    )
    required_equal_fields = (
        "semantics",
        "tensor_key",
        "tensor_shape",
        "token_strings",
        "token_ids",
        "tie_word_embeddings",
    )
    for field in required_equal_fields:
        if metadata.get(field) != expected[field]:
            raise RuntimeContractError(
                "special-token embedding metadata does not match the installed model",
                code="special_token_embeddings.metadata_mismatch",
                context={
                    "field": field,
                    "expected": expected[field],
                    "actual": metadata.get(field),
                },
            )
    optional_expected_fields = (
        "base_model_path",
        "base_config_sha256",
        "tokenizer_sha256",
    )
    for field in optional_expected_fields:
        actual_value = metadata.get(field)
        expected_value = expected[field]
        if actual_value is not None and expected_value is None:
            raise RuntimeContractError(
                "special-token embedding metadata identity requires an expected value",
                code="special_token_embeddings.identity_mismatch",
                context={
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                },
            )
        if expected_value is not None and actual_value != expected_value:
            raise RuntimeContractError(
                "special-token embedding metadata identity does not match expected base",
                code="special_token_embeddings.identity_mismatch",
                context={
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                },
            )
    if metadata.get("tensor_dtype") != _dtype_name(result.shared_embed_delta.dtype):
        raise RuntimeContractError(
            "special-token embedding metadata dtype does not match installed delta",
            code="special_token_embeddings.dtype_mismatch",
            context={
                "expected_dtype": _dtype_name(result.shared_embed_delta.dtype),
                "actual_dtype": metadata.get("tensor_dtype"),
            },
        )


def _delta_parameter_name(model: nn.Module, delta: nn.Parameter) -> str:
    for name, parameter in model.named_parameters():
        if parameter is delta:
            return name
    raise RuntimeContractError(
        "installed special-token delta parameter is not reachable from model",
        code="special_token_embeddings.delta_parameter_missing",
    )


def _parameter_name(model: nn.Module, target: nn.Parameter) -> str | None:
    for name, parameter in model.named_parameters():
        if parameter is target:
            return name
    return None


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
