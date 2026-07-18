from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from src.common.errors import RuntimeContractError
from src.config.models import (
    SpecialTokenEmbeddingGroupsConfig,
    SpecialTokenEmbeddingsConfig,
)
from src.qwen.special_token_embeddings import (
    DEFAULT_EMBED_DELTA_TENSOR_KEY,
    SPECIAL_TOKEN_EMBEDDINGS_JSON,
    SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS,
    SelectedDeltaOutputHead,
    SpecialTokenEmbeddingSourceGateEvidence,
    SpecialTokenSelection,
    build_default_special_token_selection,
    fold_special_token_embedding_delta_for_execution,
    inspect_special_token_embedding_delta_payload,
    install_special_token_embedding_deltas,
    load_inference_embedding_delta,
    load_default_special_token_embedding_source_gate_evidence,
    load_special_token_embedding_deltas,
    save_special_token_embedding_deltas,
    validate_inference_embedding_delta_identity,
)
from src.qwen.tokens import (
    DEFAULT_COORDINATE_TOKENS,
    DEFAULT_WRAPPER_TOKENS,
    QwenTokenIdentity,
)


def test_default_special_token_selection_uses_wrappers_then_coordinates() -> None:
    identity = _token_identity()
    config = SpecialTokenEmbeddingsConfig(
        groups=SpecialTokenEmbeddingGroupsConfig(
            coordinate_tokens="default_coord_0_999",
            wrapper_tokens="default_object_box_wrappers",
        )
    )

    selection = build_default_special_token_selection(config, identity)

    assert len(selection) == 1004
    assert selection.token_strings[:4] == DEFAULT_WRAPPER_TOKENS
    assert selection.token_ids[:4] == (151646, 151647, 151648, 151649)
    assert selection.token_strings[4] == "<|coord_0|>"
    assert selection.token_ids[4:] == tuple(range(151670, 152670))
    artifact = selection.to_artifact_dict()
    assert artifact["selected_token_count"] == 1004
    assert artifact["coord_token_ids_contiguous"] is True


def test_default_special_token_embedding_source_gate_loads_canonical_evidence() -> None:
    evidence = load_default_special_token_embedding_source_gate_evidence(
        Path(__file__).resolve().parents[2]
    )

    assert evidence.source_study_passed is True
    assert evidence.roundtrip_probe_passed is True
    assert evidence.probe_receipt is not None
    assert evidence.probe_receipt["ok"] is True
    assert evidence.probe_receipt["semantics"] == "additive_delta"
    assert evidence.probe_receipt["num_selected_tokens"] == 1004
    assert evidence.probe_receipt["runtime_tied_input_lm_head_identity"] is True


def test_special_token_embedding_install_requires_source_gate() -> None:
    model = TinyTiedQwenModel()
    selection = SpecialTokenSelection(token_strings=("<a>",), token_ids=(2,))

    with pytest.raises(RuntimeContractError) as missing_study:
        install_special_token_embedding_deltas(
            model,
            selection,
            source_gate=SpecialTokenEmbeddingSourceGateEvidence(
                source_study_passed=False,
                roundtrip_probe_passed=True,
                probe_receipt={"ok": True},
            ),
        )

    assert missing_study.value.code == "special_token_embeddings.source_gate_missing"

    with pytest.raises(RuntimeContractError) as missing_receipt:
        install_special_token_embedding_deltas(
            TinyTiedQwenModel(),
            selection,
            source_gate=SpecialTokenEmbeddingSourceGateEvidence(
                source_study_passed=True,
                roundtrip_probe_passed=True,
                probe_receipt=None,
            ),
        )

    assert missing_receipt.value.code == "special_token_embeddings.source_gate_missing"


@pytest.mark.parametrize(
    ("receipt_patch", "expected_code"),
    [
        (
            {"semantics": "absolute_rows"},
            "special_token_embeddings.source_gate_semantics",
        ),
        (
            {"num_selected_tokens": 2},
            "special_token_embeddings.source_gate_selected_count",
        ),
    ],
)
def test_special_token_embedding_source_gate_rejects_drifted_probe_receipts(
    receipt_patch: dict[str, object],
    expected_code: str,
) -> None:
    receipt = {
        "ok": True,
        "semantics": "additive_delta",
        "num_selected_tokens": 1,
    }
    receipt.update(receipt_patch)

    with pytest.raises(RuntimeContractError) as exc_info:
        install_special_token_embedding_deltas(
            TinyTiedQwenModel(),
            SpecialTokenSelection(token_strings=("<a>",), token_ids=(2,)),
            source_gate=SpecialTokenEmbeddingSourceGateEvidence(
                source_study_passed=True,
                roundtrip_probe_passed=True,
                probe_receipt=receipt,
            ),
        )

    assert exc_info.value.code == expected_code


def test_tied_special_token_deltas_affect_only_selected_inputs_and_logits() -> None:
    torch.manual_seed(3)
    model = TinyTiedQwenModel(vocab_size=8, hidden_size=4)
    selection = SpecialTokenSelection(token_strings=("<a>", "<b>"), token_ids=(2, 5))

    result = install_special_token_embedding_deltas(
        model,
        selection,
        source_gate=_source_gate(selected_count=2),
    )

    assert result.receipt.tie_word_embeddings is True
    assert result.receipt.delta_parameter_names == ("embed_tokens.shared_embed_delta",)
    assert result.shared_embed_delta.shape == (2, 4)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert not result.input_wrapper.base.weight.requires_grad
    assert not result.output_wrapper.base.weight.requires_grad
    trainable = [param for param in result.model.parameters() if param.requires_grad]
    assert trainable == [result.shared_embed_delta]

    with torch.no_grad():
        result.shared_embed_delta.copy_(
            torch.tensor(
                [[0.25, -0.5, 0.75, 1.0], [-1.0, 0.5, 0.125, -0.75]],
                dtype=result.shared_embed_delta.dtype,
            )
        )

    input_ids = torch.tensor([[2, 1, 5]], dtype=torch.long)
    base_input = result.input_wrapper.base(input_ids).detach()
    wrapped_input = model.get_input_embeddings()(input_ids).detach()
    input_diff = wrapped_input - base_input
    assert torch.allclose(input_diff[0, 0], result.shared_embed_delta[0])
    assert torch.allclose(input_diff[0, 1], torch.zeros(4))
    assert torch.allclose(input_diff[0, 2], result.shared_embed_delta[1])

    hidden = torch.tensor([[0.5, 1.0, -0.5, 0.25]], dtype=torch.float32)
    base_logits = result.output_wrapper.base(hidden).detach()
    wrapped_logits = model.get_output_embeddings()(hidden).detach()
    logits_diff = wrapped_logits - base_logits
    expected = hidden @ result.shared_embed_delta.t()
    assert torch.allclose(logits_diff[:, [2, 5]], expected)
    non_selected = torch.ones(8, dtype=torch.bool)
    non_selected[[2, 5]] = False
    assert torch.allclose(
        logits_diff[:, non_selected],
        torch.zeros_like(logits_diff[:, non_selected]),
    )

    loss = model.get_output_embeddings()(model.get_input_embeddings()(input_ids)).sum()
    loss.backward()
    assert result.shared_embed_delta.grad is not None
    assert result.shared_embed_delta.grad.abs().sum().item() > 0
    assert result.input_wrapper.base.weight.grad is None
    assert result.output_wrapper.base.weight.grad is None


def test_special_token_embedding_delta_owner_dtype_is_fp32_for_bf16_base() -> None:
    model = TinyTiedQwenModel(vocab_size=8, hidden_size=4).to(dtype=torch.bfloat16)
    selection = SpecialTokenSelection(token_strings=("<a>", "<b>"), token_ids=(2, 5))

    result = install_special_token_embedding_deltas(
        model,
        selection,
        source_gate=_source_gate(selected_count=2),
    )

    assert result.input_wrapper.base.weight.dtype == torch.bfloat16
    assert result.output_wrapper.base.weight.dtype == torch.bfloat16
    assert result.shared_embed_delta.dtype == torch.float32
    assert result.receipt.delta_dtype == "float32"
    assert (
        result.receipt.to_metadata_dict(
            base_model_path=None,
            base_config_sha256=None,
            tokenizer_sha256=None,
        )["tensor_dtype"]
        == "float32"
    )


def test_selected_delta_output_head_avoids_full_logits_clone() -> None:
    torch.manual_seed(13)
    base = nn.Linear(4, 128, bias=False)
    selection = SpecialTokenSelection(token_strings=("<a>", "<b>"), token_ids=(3, 99))
    delta = nn.Parameter(torch.randn(2, 4) * 0.01)
    wrapper = SelectedDeltaOutputHead(base, selection, delta)
    hidden = torch.randn(2, 5, 4, requires_grad=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        logits = wrapper(hidden)

    assert logits.shape == (2, 5, 128)
    assert not any(event.key == "aten::clone" for event in prof.key_averages())


def test_special_token_embedding_install_preserves_existing_trainable_adapters() -> (
    None
):
    model = TinyTiedQwenWithAdapter(vocab_size=8, hidden_size=4)
    selection = SpecialTokenSelection(token_strings=("<a>",), token_ids=(2,))

    result = install_special_token_embedding_deltas(
        model,
        selection,
        source_gate=_source_gate(selected_count=1),
    )

    trainable_names = {
        name
        for name, parameter in result.model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_names == {
        "adapter_weight",
        "embed_tokens.shared_embed_delta",
    }
    assert result.receipt.delta_parameter_names == ("embed_tokens.shared_embed_delta",)


def test_special_token_embedding_install_rejects_selected_id_outside_vocab() -> None:
    model = TinyTiedQwenModel(vocab_size=8, hidden_size=4)
    selection = SpecialTokenSelection(token_strings=("<a>",), token_ids=(8,))

    with pytest.raises(RuntimeContractError) as exc_info:
        install_special_token_embedding_deltas(
            model,
            selection,
            source_gate=_source_gate(selected_count=1),
        )

    assert exc_info.value.code == "special_token_embeddings.token_id_out_of_range"


def test_special_token_embedding_compact_payload_round_trips(tmp_path: Path) -> None:
    selection = SpecialTokenSelection(token_strings=("<a>", "<b>"), token_ids=(2, 5))
    result = install_special_token_embedding_deltas(
        TinyTiedQwenModel(vocab_size=8, hidden_size=4),
        selection,
        source_gate=_source_gate(selected_count=2),
    )
    with torch.no_grad():
        result.shared_embed_delta.copy_(
            torch.tensor(
                [[0.5, 0.25, -0.125, 0.75], [-0.5, 0.75, 0.25, -0.25]],
                dtype=result.shared_embed_delta.dtype,
            )
        )

    payload = save_special_token_embedding_deltas(
        result,
        tmp_path,
        base_model_path=Path("/models/qwen-base"),
        base_config_sha256="config-sha",
        tokenizer_sha256="tokenizer-sha",
    )

    assert payload.tensor_path == tmp_path / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS
    assert payload.metadata_path == tmp_path / SPECIAL_TOKEN_EMBEDDINGS_JSON
    metadata = json.loads(payload.metadata_path.read_text(encoding="utf-8"))
    assert metadata["semantics"] == "additive_delta"
    assert metadata["tensor_key"] == DEFAULT_EMBED_DELTA_TENSOR_KEY
    assert metadata["token_ids"] == [2, 5]
    assert metadata["tie_word_embeddings"] is True
    assert metadata["base_model_path"] == "/models/qwen-base"

    reloaded = install_special_token_embedding_deltas(
        TinyTiedQwenModel(vocab_size=8, hidden_size=4),
        selection,
        source_gate=_source_gate(selected_count=2),
    )
    load_receipt = load_special_token_embedding_deltas(
        reloaded,
        tmp_path,
        expected_base_model_path=Path("/models/qwen-base"),
        expected_base_config_sha256="config-sha",
        expected_tokenizer_sha256="tokenizer-sha",
    )

    assert load_receipt.loaded is True
    assert torch.allclose(reloaded.shared_embed_delta, result.shared_embed_delta)
    input_ids = torch.tensor([[2, 1]], dtype=torch.long)
    input_diff = reloaded.input_wrapper(input_ids) - reloaded.input_wrapper.base(
        input_ids
    )
    assert torch.allclose(input_diff[0, 0], result.shared_embed_delta[0])
    assert torch.allclose(input_diff[0, 1], torch.zeros(4))


def test_execution_delta_identity_is_path_independent_and_binds_tensor_bytes(
    tmp_path: Path,
) -> None:
    payload_dir, _ = _write_execution_delta(tmp_path / "original")
    identity = inspect_special_token_embedding_delta_payload(
        payload_dir,
        expected_base_model_path=Path("/models/qwen-base"),
        expected_base_config_sha256="config-sha",
        expected_tokenizer_sha256="tokenizer-sha",
    )
    copied_dir = tmp_path / "copied"
    shutil.copytree(payload_dir, copied_dir)
    copied = inspect_special_token_embedding_delta_payload(copied_dir)

    assert identity["kind"] == "special_token_embedding_delta"
    assert identity["fingerprint"] == copied["fingerprint"]
    assert identity["root"] != copied["root"]
    assert identity["semantic_identity"]["tensor_key"] == (
        DEFAULT_EMBED_DELTA_TENSOR_KEY
    )
    assert identity["semantic_identity"]["tensor_dtype"] == "float32"

    tensor_path = copied_dir / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS
    changed = torch.tensor(
        [[0.75, -0.5, 0.25, 1.0], [-0.25, 0.5, -1.0, 0.125]],
        dtype=torch.float32,
    )
    save_file({DEFAULT_EMBED_DELTA_TENSOR_KEY: changed}, tensor_path)
    changed_identity = inspect_special_token_embedding_delta_payload(copied_dir)
    assert changed_identity["fingerprint"] != identity["fingerprint"]


def test_fold_execution_delta_adds_once_in_target_dtype_and_preserves_tie(
    tmp_path: Path,
) -> None:
    payload_dir, delta = _write_execution_delta(tmp_path / "payload")
    identity = inspect_special_token_embedding_delta_payload(payload_dir)
    model = TinyTiedQwenModel(vocab_size=8, hidden_size=4).to(torch.bfloat16)
    tied_weight = model.get_input_embeddings().weight
    before = tied_weight.detach().clone()
    expected = before.clone()
    expected.index_add_(
        0,
        torch.tensor([2, 5]),
        delta.to(dtype=torch.bfloat16),
    )

    receipt = fold_special_token_embedding_delta_for_execution(
        model,
        payload_dir,
        expected_identity=identity,
    )

    assert torch.equal(tied_weight, expected)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert receipt["source_dtype"] == "float32"
    assert receipt["target_dtype"] == "bfloat16"
    assert receipt["row_addition_count"] == 1


def test_execution_delta_accepts_bfloat16_source_and_normalizes_for_fold(
    tmp_path: Path,
) -> None:
    payload_dir, delta = _write_execution_delta(tmp_path / "payload")
    tensor_path = payload_dir / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS
    save_file(
        {DEFAULT_EMBED_DELTA_TENSOR_KEY: delta.to(torch.bfloat16)},
        tensor_path,
    )
    metadata_path = payload_dir / SPECIAL_TOKEN_EMBEDDINGS_JSON
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tensor_dtype"] = "bfloat16"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    identity = inspect_special_token_embedding_delta_payload(payload_dir)
    model = TinyTiedQwenModel(vocab_size=8, hidden_size=4).to(torch.bfloat16)
    receipt = fold_special_token_embedding_delta_for_execution(
        model,
        payload_dir,
        expected_identity=identity,
    )

    assert receipt["source_dtype"] == "bfloat16"
    assert receipt["runtime_delta_dtype"] == "float32"
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert receipt["tied_input_output_storage"] is True
    assert receipt["wrapper_modules"] == []


def test_fold_execution_delta_rejects_wrappers_and_out_of_range_ids(
    tmp_path: Path,
) -> None:
    payload_dir, _ = _write_execution_delta(tmp_path / "payload")
    wrapped = install_special_token_embedding_deltas(
        TinyTiedQwenModel(vocab_size=8, hidden_size=4),
        SpecialTokenSelection(token_strings=("<a>", "<b>"), token_ids=(2, 5)),
        source_gate=_source_gate(selected_count=2),
    ).model

    with pytest.raises(RuntimeContractError) as wrapper_error:
        fold_special_token_embedding_delta_for_execution(wrapped, payload_dir)
    assert wrapper_error.value.code == (
        "special_token_embeddings.execution_wrapper_residue"
    )

    with pytest.raises(RuntimeContractError) as range_error:
        fold_special_token_embedding_delta_for_execution(
            TinyTiedQwenModel(vocab_size=5, hidden_size=4),
            payload_dir,
        )
    assert range_error.value.code == (
        "special_token_embeddings.execution_token_id_out_of_range"
    )


def test_execution_delta_inspection_rejects_duplicate_token_ids(
    tmp_path: Path,
) -> None:
    payload_dir, _ = _write_execution_delta(tmp_path / "payload")
    metadata_path = payload_dir / SPECIAL_TOKEN_EMBEDDINGS_JSON
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["token_ids"] = [2, 2]
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeContractError) as exc_info:
        inspect_special_token_embedding_delta_payload(payload_dir)

    assert exc_info.value.code == "special_token_embeddings.execution_token_ids"


@pytest.mark.parametrize(
    "load_kwargs",
    [
        {},
        {"expected_base_model_path": Path("/models/other-qwen-base")},
        {"expected_base_config_sha256": "other-config-sha"},
        {"expected_tokenizer_sha256": "other-tokenizer-sha"},
    ],
)
def test_special_token_embedding_load_requires_matching_identity(
    tmp_path: Path,
    load_kwargs: dict[str, object],
) -> None:
    selection = SpecialTokenSelection(token_strings=("<a>",), token_ids=(2,))
    result = install_special_token_embedding_deltas(
        TinyTiedQwenModel(vocab_size=8, hidden_size=4),
        selection,
        source_gate=_source_gate(selected_count=1),
    )
    save_special_token_embedding_deltas(
        result,
        tmp_path,
        base_model_path=Path("/models/qwen-base"),
        base_config_sha256="config-sha",
        tokenizer_sha256="tokenizer-sha",
    )
    reloaded = install_special_token_embedding_deltas(
        TinyTiedQwenModel(vocab_size=8, hidden_size=4),
        selection,
        source_gate=_source_gate(selected_count=1),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        load_special_token_embedding_deltas(reloaded, tmp_path, **load_kwargs)

    assert exc_info.value.code == "special_token_embeddings.identity_mismatch"


def test_special_token_embedding_load_rejects_tensor_dtype_mismatch(
    tmp_path: Path,
) -> None:
    selection = SpecialTokenSelection(token_strings=("<a>",), token_ids=(2,))
    result = install_special_token_embedding_deltas(
        TinyTiedQwenModel(vocab_size=8, hidden_size=4),
        selection,
        source_gate=_source_gate(selected_count=1),
    )
    metadata = result.receipt.to_metadata_dict(
        base_model_path=Path("/models/qwen-base"),
        base_config_sha256="config-sha",
        tokenizer_sha256="tokenizer-sha",
    )
    (tmp_path / SPECIAL_TOKEN_EMBEDDINGS_JSON).write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    save_file(
        {DEFAULT_EMBED_DELTA_TENSOR_KEY: torch.zeros((1, 4), dtype=torch.float64)},
        str(tmp_path / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        load_special_token_embedding_deltas(
            result,
            tmp_path,
            expected_base_model_path=Path("/models/qwen-base"),
            expected_base_config_sha256="config-sha",
            expected_tokenizer_sha256="tokenizer-sha",
        )

    assert exc_info.value.code == "special_token_embeddings.dtype_mismatch"


def test_special_token_embedding_load_converts_self_consistent_bf16_payload_to_fp32(
    tmp_path: Path,
) -> None:
    selection = SpecialTokenSelection(token_strings=("<a>",), token_ids=(2,))
    result = install_special_token_embedding_deltas(
        TinyTiedQwenModel(vocab_size=8, hidden_size=4),
        selection,
        source_gate=_source_gate(selected_count=1),
    )
    expected = torch.tensor([[0.125, -0.5, 1.75, 3.0]], dtype=torch.bfloat16)
    metadata = result.receipt.to_metadata_dict(
        base_model_path=Path("/models/qwen-base"),
        base_config_sha256="config-sha",
        tokenizer_sha256="tokenizer-sha",
    )
    metadata["tensor_dtype"] = "bfloat16"
    (tmp_path / SPECIAL_TOKEN_EMBEDDINGS_JSON).write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    save_file(
        {DEFAULT_EMBED_DELTA_TENSOR_KEY: expected},
        str(tmp_path / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS),
    )

    receipt = load_special_token_embedding_deltas(
        result,
        tmp_path,
        expected_base_model_path=Path("/models/qwen-base"),
        expected_base_config_sha256="config-sha",
        expected_tokenizer_sha256="tokenizer-sha",
    )

    assert result.shared_embed_delta.dtype == torch.float32
    assert torch.equal(result.shared_embed_delta, expected.float())
    assert receipt.tensor_dtype == "float32"
    assert receipt.source_tensor_dtype == "bfloat16"
    assert receipt.runtime_tensor_dtype == "float32"
    assert receipt.to_artifact_dict()["source_tensor_dtype"] == "bfloat16"
    assert receipt.to_artifact_dict()["runtime_tensor_dtype"] == "float32"


def test_special_token_embedding_load_rejects_full_embedding_payload(
    tmp_path: Path,
) -> None:
    selection = SpecialTokenSelection(token_strings=("<a>",), token_ids=(2,))
    result = install_special_token_embedding_deltas(
        TinyTiedQwenModel(vocab_size=8, hidden_size=4),
        selection,
        source_gate=_source_gate(selected_count=1),
    )
    metadata = result.receipt.to_metadata_dict(
        base_model_path=Path("/models/qwen-base"),
        base_config_sha256="config-sha",
        tokenizer_sha256="tokenizer-sha",
    )
    (tmp_path / SPECIAL_TOKEN_EMBEDDINGS_JSON).write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    save_file(
        {
            DEFAULT_EMBED_DELTA_TENSOR_KEY: torch.zeros((1, 4)),
            "embed_tokens.weight": torch.zeros((8, 4)),
        },
        str(tmp_path / SPECIAL_TOKEN_EMBEDDINGS_SAFE_TENSORS),
    )

    with pytest.raises(RuntimeContractError) as exc_info:
        load_special_token_embedding_deltas(
            result,
            tmp_path,
            expected_base_model_path=Path("/models/qwen-base"),
            expected_base_config_sha256="config-sha",
            expected_tokenizer_sha256="tokenizer-sha",
        )

    assert exc_info.value.code == "special_token_embeddings.unexpected_tensor_keys"


def test_inference_embedding_delta_identity_accepts_matching_metadata(
    tmp_path: Path,
) -> None:
    metadata_path = _write_inference_delta_metadata(tmp_path)

    receipt = validate_inference_embedding_delta_identity(
        config=_delta_config(tmp_path),
        qwen=_qwen_identity_context(),
    )

    assert receipt["status"] == "validated"
    assert receipt["metadata_path"] == str(metadata_path)
    assert receipt["base_model_path"] == "/models/qwen-base"
    assert receipt["metadata"]["base_config_sha256"] == "base-config-sha"
    assert receipt["metadata"]["tokenizer_sha256"] == "tokenizer-sha"


def test_inference_embedding_delta_load_installs_wrappers_and_payload(
    tmp_path: Path,
) -> None:
    qwen = _qwen_identity_context(
        model=TinyTiedQwenModel(vocab_size=152670, hidden_size=4)
    )
    selection = build_default_special_token_selection(
        SpecialTokenEmbeddingsConfig(
            groups=SpecialTokenEmbeddingGroupsConfig(
                coordinate_tokens="default_coord_0_999",
                wrapper_tokens="default_object_box_wrappers",
            )
        ),
        qwen.token_identity,
    )
    payload_result = install_special_token_embedding_deltas(
        TinyTiedQwenModel(vocab_size=152670, hidden_size=4),
        selection,
        source_gate=_source_gate(selected_count=len(selection)),
    )
    with torch.no_grad():
        payload_result.shared_embed_delta.fill_(0.125)
    save_special_token_embedding_deltas(
        payload_result,
        tmp_path,
        base_model_path=Path("/models/qwen-base"),
        base_config_sha256="base-config-sha",
        tokenizer_sha256="tokenizer-sha",
    )

    receipt = load_inference_embedding_delta(config=_delta_config(tmp_path), qwen=qwen)

    assert receipt["status"] == "loaded"
    assert receipt["identity"]["status"] == "validated"
    assert receipt["load"]["loaded"] is True
    assert qwen.model.get_input_embeddings().selection.token_ids == selection.token_ids
    assert (
        qwen.model.get_output_embeddings().selection.token_strings
        == selection.token_strings
    )
    assert torch.allclose(
        qwen.model.get_input_embeddings().shared_embed_delta,
        torch.full((len(selection), 4), 0.125),
    )


def test_inference_embedding_delta_load_requires_loaded_model(tmp_path: Path) -> None:
    _write_inference_delta_metadata(tmp_path)

    with pytest.raises(RuntimeContractError) as exc_info:
        load_inference_embedding_delta(
            config=_delta_config(tmp_path),
            qwen=_qwen_identity_context(model=None),
        )

    assert exc_info.value.code == "special_token_embeddings.model_not_loaded"


def test_inference_embedding_delta_identity_rejects_token_id_mismatch(
    tmp_path: Path,
) -> None:
    metadata = _inference_delta_metadata()
    metadata["token_ids"] = [*metadata["token_ids"]]
    metadata["token_ids"][-1] += 1
    _write_inference_delta_metadata(tmp_path, metadata=metadata)

    with pytest.raises(RuntimeContractError) as exc_info:
        validate_inference_embedding_delta_identity(
            config=_delta_config(tmp_path),
            qwen=_qwen_identity_context(),
        )

    assert exc_info.value.code == "special_token_embeddings.identity_mismatch"
    assert exc_info.value.context["field"] == "token_ids"


@pytest.mark.parametrize("field", ["base_config_sha256", "tokenizer_sha256"])
def test_inference_embedding_delta_identity_requires_sha_metadata(
    tmp_path: Path,
    field: str,
) -> None:
    metadata = _inference_delta_metadata()
    del metadata[field]
    _write_inference_delta_metadata(tmp_path, metadata=metadata)

    with pytest.raises(RuntimeContractError) as exc_info:
        validate_inference_embedding_delta_identity(
            config=_delta_config(tmp_path),
            qwen=_qwen_identity_context(),
        )

    assert exc_info.value.code == "special_token_embeddings.inference_identity_missing"
    assert field in exc_info.value.context["missing_fields"]


@pytest.mark.parametrize(
    ("runtime_patch", "field"),
    [
        ({"base_config_sha256": "other-base-config-sha"}, "base_config_sha256"),
        ({"tokenizer_sha256": "other-tokenizer-sha"}, "tokenizer_sha256"),
    ],
)
def test_inference_embedding_delta_identity_rejects_wrong_runtime_sha(
    tmp_path: Path,
    runtime_patch: dict[str, str],
    field: str,
) -> None:
    _write_inference_delta_metadata(tmp_path)
    runtime = _qwen_identity_context(**runtime_patch)

    with pytest.raises(RuntimeContractError) as exc_info:
        validate_inference_embedding_delta_identity(
            config=_delta_config(tmp_path),
            qwen=runtime,
        )

    assert exc_info.value.code == "special_token_embeddings.identity_mismatch"
    assert exc_info.value.context["field"] == field


def test_inference_embedding_delta_identity_rejects_null_sha_metadata_with_runtime_sha(
    tmp_path: Path,
) -> None:
    metadata = _inference_delta_metadata()
    metadata["base_config_sha256"] = None
    metadata["tokenizer_sha256"] = None
    _write_inference_delta_metadata(tmp_path, metadata=metadata)

    with pytest.raises(RuntimeContractError) as exc_info:
        validate_inference_embedding_delta_identity(
            config=_delta_config(tmp_path),
            qwen=_qwen_identity_context(),
        )

    assert exc_info.value.code == "special_token_embeddings.identity_mismatch"
    assert exc_info.value.context["field"] == "base_config_sha256"


@pytest.mark.parametrize("field", ["base_config_sha256", "tokenizer_sha256"])
def test_inference_embedding_delta_identity_requires_runtime_sha(
    tmp_path: Path,
    field: str,
) -> None:
    _write_inference_delta_metadata(tmp_path)
    runtime = _qwen_identity_context(**{field: None})

    with pytest.raises(RuntimeContractError) as exc_info:
        validate_inference_embedding_delta_identity(
            config=_delta_config(tmp_path),
            qwen=runtime,
        )

    assert exc_info.value.code == "special_token_embeddings.runtime_identity_missing"
    assert exc_info.value.context["field"] == field


class TinyTiedQwenModel(nn.Module):
    def __init__(self, *, vocab_size: int = 8, hidden_size: int = 4) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, output_embeddings: nn.Module) -> None:
        self.lm_head = output_embeddings


class TinyTiedQwenWithAdapter(TinyTiedQwenModel):
    def __init__(self, *, vocab_size: int = 8, hidden_size: int = 4) -> None:
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size)
        self.adapter_weight = nn.Parameter(torch.ones(hidden_size))


def _write_execution_delta(path: Path) -> tuple[Path, torch.Tensor]:
    selection = SpecialTokenSelection(
        token_strings=("<a>", "<b>"),
        token_ids=(2, 5),
    )
    installed = install_special_token_embedding_deltas(
        TinyTiedQwenModel(vocab_size=8, hidden_size=4),
        selection,
        source_gate=_source_gate(selected_count=2),
    )
    delta = torch.tensor(
        [[0.5, 0.25, -0.125, 0.75], [-0.5, 0.75, 0.25, -0.25]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        installed.shared_embed_delta.copy_(delta)
    save_special_token_embedding_deltas(
        installed,
        path,
        base_model_path=Path("/models/qwen-base"),
        base_config_sha256="config-sha",
        tokenizer_sha256="tokenizer-sha",
    )
    return path, delta


def _source_gate(*, selected_count: int) -> SpecialTokenEmbeddingSourceGateEvidence:
    return SpecialTokenEmbeddingSourceGateEvidence(
        source_study_passed=True,
        roundtrip_probe_passed=True,
        probe_receipt={
            "ok": True,
            "semantics": "additive_delta",
            "num_selected_tokens": selected_count,
            "runtime_tied_input_lm_head_identity": True,
            "payload": {
                "safetensors": "special_token_embeddings.safetensors",
                "metadata": "special_token_embeddings.json",
            },
        },
    )


def _token_identity() -> QwenTokenIdentity:
    return QwenTokenIdentity(
        required_tokens=(*DEFAULT_WRAPPER_TOKENS, *DEFAULT_COORDINATE_TOKENS),
        wrapper_token_ids={
            token: 151646 + index for index, token in enumerate(DEFAULT_WRAPPER_TOKENS)
        },
        coordinate_token_ids=tuple(range(151670, 152670)),
        im_end_newline_text="<|im_end|>\n",
        im_end_token_ids=(151645,),
        newline_token_ids=(198,),
        im_end_newline_token_ids=(151645, 198),
        tokenizer_vocab_size=152670,
    )


def _inference_delta_metadata() -> dict[str, object]:
    identity = _token_identity()
    selection = build_default_special_token_selection(
        SpecialTokenEmbeddingsConfig(
            groups=SpecialTokenEmbeddingGroupsConfig(
                coordinate_tokens="default_coord_0_999",
                wrapper_tokens="default_object_box_wrappers",
            )
        ),
        identity,
    )
    return {
        "semantics": "additive_delta",
        "tensor_key": DEFAULT_EMBED_DELTA_TENSOR_KEY,
        "tensor_shape": [len(selection), 4],
        "tensor_dtype": "torch.float32",
        "token_strings": list(selection.token_strings),
        "token_ids": list(selection.token_ids),
        "base_model_path": "/models/qwen-base",
        "base_config_sha256": "base-config-sha",
        "tokenizer_sha256": "tokenizer-sha",
        "tie_word_embeddings": True,
    }


def _write_inference_delta_metadata(
    path: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / SPECIAL_TOKEN_EMBEDDINGS_JSON
    metadata_path.write_text(
        json.dumps(metadata or _inference_delta_metadata(), sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


def _delta_config(path: Path) -> SimpleNamespace:
    return SimpleNamespace(embedding_delta=SimpleNamespace(path=str(path)))


def _qwen_identity_context(**overrides: object) -> SimpleNamespace:
    payload = {
        "base_model_path": "/models/qwen-base",
        "base_config_sha256": "base-config-sha",
        "tokenizer_sha256": "tokenizer-sha",
        "token_identity": _token_identity(),
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)
