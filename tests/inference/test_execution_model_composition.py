from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.common.errors import RuntimeContractError
from src.inference.execution_model import (
    bind_execution_model_composition,
    load_execution_model_receipt,
    resolve_execution_model,
    validate_execution_model_receipt,
)
from src.inference.execution_model_composition import (
    build_execution_model_composition_receipt,
    compare_execution_models,
    validate_execution_model_composition_receipt,
    write_execution_model_composition_receipt,
)


def _write_snapshot(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "tie_word_embeddings": True,
                "dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    save_file(
        {
            "model.language_model.embed_tokens.weight": torch.zeros(
                (2, 2), dtype=torch.bfloat16
            )
        },
        root / "model.safetensors",
    )
    return root


def _target_identity() -> dict[str, object]:
    return {
        "target_count": 1,
        "targets": [
            {
                "target_name": "model.language_model.layers.0.self_attn.q_proj",
                "shape": [4, 4],
                "dtype": "torch.bfloat16",
                "sha256": "a" * 64,
            }
        ],
        "fingerprint": "b" * 64,
    }


def _comparison(*, native_dtype: str = "torch.bfloat16") -> dict[str, object]:
    target_identity = _target_identity()
    selected_rows_sha256 = "c" * 64
    return {
        "composition_checks": {
            "prompt_ids": True,
            "selected_rows_target_dtype": True,
            "dynamic_tied_weights": True,
            "materialized_tied_weights": True,
            "merged_target_weights": True,
        },
        "behavior_checks": {
            "greedy_generated_ids_match": False,
            "full_vocab_within_reference_tolerance": False,
            "selected_vocab_within_reference_tolerance": False,
        },
        "merged_target_weight_identity": {
            "expected": {
                "target_count": target_identity["target_count"],
                "fingerprint": target_identity["fingerprint"],
            },
            "observed": {
                "target_count": target_identity["target_count"],
                "fingerprint": target_identity["fingerprint"],
            },
        },
        "selected_row_identity": {
            "dynamic_effective_sha256": selected_rows_sha256,
            "materializer_folded_sha256": selected_rows_sha256,
            "reloaded_materialized_sha256": selected_rows_sha256,
        },
        "full_vocab": {
            "allclose": False,
            "rtol": 1e-4,
            "atol": 5e-3,
            "max_abs_diff": 0.001,
            "max_rel_diff": 0.0001,
        },
        "selected_vocab": {
            "allclose": False,
            "rtol": 1e-4,
            "atol": 2e-3,
            "max_abs_diff": 0.0005,
            "max_rel_diff": 0.0001,
        },
        "compared_positions": [0, 1],
        "full_vocab_shape": [2, 151936],
        "selected_vocab_shape": [2, 1004],
        "dtypes": {
            "dynamic_native_logits": native_dtype,
            "materialized_native_logits": native_dtype,
            "comparison_logits": "torch.float32",
            "dynamic_selected_rows": native_dtype,
            "materialized_selected_rows": native_dtype,
        },
        "dynamic_generated_ids": [1, 2],
        "materialized_generated_ids": [1, 3],
    }


def _materialized_execution_model(tmp_path: Path) -> dict[str, object]:
    target_identity = _target_identity()

    def materialize(snapshot_root: Path) -> dict[str, object]:
        _write_snapshot(snapshot_root)
        return {
            "adapter_merge": {
                "merge": {"target_weight_identity": target_identity}
            },
            "embedding_delta_fold": {
                "selected_rows_after_sha256": "c" * 64
            },
        }

    return resolve_execution_model(
        base_model_path=_write_snapshot(tmp_path / "base"),
        target_dtype="bf16",
        adapter_identity={"kind": "adapter", "fingerprint": "a" * 64},
        embedding_delta_identity={"kind": "delta", "fingerprint": "d" * 64},
        cache_root=tmp_path / "cache",
        materialize_snapshot=materialize,
    )


def _fixture() -> dict[str, object]:
    return {
        "row_id": "fixture-0",
        "row_index": 0,
        "input_jsonl_sha256": "1" * 64,
        "prompt_ids_sha256": "2" * 64,
        "dynamic_executed_prompt_ids_sha256": "2" * 64,
        "materialized_executed_prompt_ids_sha256": "2" * 64,
        "processor_fingerprint": "3" * 64,
        "image_file_sha256": "4" * 64,
        "executed_media_sha256": "5" * 64,
        "generation_fingerprint": "6" * 64,
        "max_new_tokens": 64,
    }


def _probe_identity() -> dict[str, object]:
    path = Path("scripts/probes/coordexp_swift/execution_model_composition.py")
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _config_identity() -> dict[str, object]:
    return {
        "fingerprint": "7" * 64,
        "entry_config_path": "/config.yaml",
        "sources": [{"path": "/base.yaml", "sha256": "8" * 64}],
    }


def _build_receipt(execution_model: dict[str, object]) -> dict[str, object]:
    return build_execution_model_composition_receipt(
        execution_model=execution_model,
        fixture_identity=_fixture(),
        probe_identity=_probe_identity(),
        resolved_config_identity=_config_identity(),
        comparison=_comparison(),
    )


def test_composition_receipt_binds_execution_identity(
    tmp_path: Path,
) -> None:
    execution_model = _materialized_execution_model(tmp_path)
    composition = _build_receipt(execution_model)
    assert validate_execution_model_composition_receipt(
        composition,
        execution_model=execution_model,
    ) == composition

    bound = bind_execution_model_composition(execution_model, composition)
    assert bound["composition_fidelity"]["digest"] == composition["digest"]
    assert validate_execution_model_receipt(bound) == bound


def test_clean_cache_binds_matching_durable_composition_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.inference import execution_model as execution_model_module

    execution_model = _materialized_execution_model(tmp_path)
    composition = _build_receipt(execution_model)
    durable_root = tmp_path / "durable-qualification"
    durable_root.mkdir()
    durable_path = durable_root / (
        f"execution-model-composition-{execution_model['composition_key']}.json"
    )
    write_execution_model_composition_receipt(durable_path, composition)
    monkeypatch.setattr(
        execution_model_module,
        "DURABLE_COMPOSITION_RECEIPT_ROOT",
        durable_root,
    )

    loaded = load_execution_model_receipt(execution_model["receipt_path"])

    assert loaded["composition_fidelity"]["digest"] == composition["digest"]
    assert loaded["composition_fidelity"]["path"] == str(durable_path.resolve())


def test_composition_receipt_accepts_fp32_execution_evidence(
    tmp_path: Path,
) -> None:
    execution_model = _materialized_execution_model(tmp_path)
    execution_model["target_dtype"] = "fp32"
    composition = build_execution_model_composition_receipt(
        execution_model=execution_model,
        fixture_identity=_fixture(),
        probe_identity=_probe_identity(),
        resolved_config_identity=_config_identity(),
        comparison=_comparison(native_dtype="torch.float32"),
    )

    assert composition["target_dtype"] == "fp32"
    assert composition["comparison"]["dtypes"]["comparison_logits"] == (
        "torch.float32"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("composition_key",), "changed"),
        (("snapshot_fingerprint",), "changed"),
        (("source_fingerprints", "base"), "changed"),
        (("target_dtype",), "fp16"),
        (("algorithm_version",), "changed"),
        (("package_versions", "torch"), "changed"),
        (("fixture_identity", "image_sha256"), "changed"),
        (("comparison", "composition_checks", "prompt_ids"), False),
        (("comparison", "selected_vocab", "allclose"), True),
        (("thresholds", "selected_vocab", "atol"), 1.0),
    ],
)
def test_composition_receipt_mutation_fails(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    execution_model = _materialized_execution_model(tmp_path)
    composition = _build_receipt(execution_model)
    mutated = copy.deepcopy(composition)
    owner = mutated
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value

    with pytest.raises(RuntimeContractError):
        validate_execution_model_composition_receipt(
            mutated,
            execution_model=execution_model,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("merged", "0" * 64), ("selected", "1" * 64)],
)
def test_composition_builder_rejects_identity_detached_from_materializer(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    execution_model = _materialized_execution_model(tmp_path)
    comparison = _comparison()
    if field == "merged":
        comparison["merged_target_weight_identity"]["expected"]["fingerprint"] = value
        comparison["merged_target_weight_identity"]["observed"]["fingerprint"] = value
    else:
        for key in comparison["selected_row_identity"]:
            comparison["selected_row_identity"][key] = value

    with pytest.raises(RuntimeContractError):
        build_execution_model_composition_receipt(
            execution_model=execution_model,
            fixture_identity=_fixture(),
            probe_identity=_probe_identity(),
            resolved_config_identity=_config_identity(),
            comparison=comparison,
        )


@pytest.mark.parametrize(
    "field",
    [
        "row_id",
        "row_index",
        "input_jsonl_sha256",
        "prompt_ids_sha256",
        "dynamic_executed_prompt_ids_sha256",
        "materialized_executed_prompt_ids_sha256",
        "processor_fingerprint",
        "image_file_sha256",
        "executed_media_sha256",
        "generation_fingerprint",
        "max_new_tokens",
    ],
)
def test_composition_builder_rejects_incomplete_fixture_identity(
    tmp_path: Path,
    field: str,
) -> None:
    execution_model = _materialized_execution_model(tmp_path)
    fixture = _fixture()
    del fixture[field]

    with pytest.raises(RuntimeContractError):
        build_execution_model_composition_receipt(
            execution_model=execution_model,
            fixture_identity=fixture,
            probe_identity=_probe_identity(),
            resolved_config_identity=_config_identity(),
            comparison=_comparison(),
        )


def test_composition_comparison_rejects_independent_processor_prompt_drift() -> None:
    with pytest.raises(RuntimeContractError) as exc_info:
        compare_execution_models(
            dynamic_model=object(),
            materialized_model=object(),
            dynamic_native_inputs={"input_ids": torch.tensor([[1, 2]])},
            materialized_native_inputs={"input_ids": torch.tensor([[1, 3]])},
            selected_token_ids=[1],
            generation_kwargs={},
            expected_merged_target_identity=_target_identity(),
            expected_folded_selected_rows_sha256="c" * 64,
        )

    assert (
        exc_info.value.code
        == "inference.execution_model_composition_prompt_mismatch"
    )
