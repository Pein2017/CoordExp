from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.common.errors import RuntimeContractError
from src.inference.execution_model import (
    bind_execution_model_parity,
    resolve_execution_model,
    validate_execution_model_receipt,
)
from src.inference.execution_model_parity import (
    build_execution_model_parity_receipt,
    validate_execution_model_parity_receipt,
)


def _write_snapshot(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_vl", "tie_word_embeddings": True}),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def _comparison() -> dict[str, object]:
    return {
        "exact_checks": {
            "prompt_ids": True,
            "greedy_generated_ids": True,
            "selected_rows_target_dtype": True,
            "dynamic_tied_weights": True,
            "materialized_tied_weights": True,
        },
        "full_vocab": {
            "allclose": True,
            "rtol": 1e-4,
            "atol": 5e-3,
            "max_abs_diff": 0.001,
            "max_rel_diff": 0.0001,
        },
        "selected_vocab": {
            "allclose": True,
            "rtol": 1e-4,
            "atol": 2e-3,
            "max_abs_diff": 0.0005,
            "max_rel_diff": 0.0001,
        },
        "compared_positions": [0, 1],
        "full_vocab_shape": [2, 151936],
        "selected_vocab_shape": [2, 1004],
        "dtypes": {"dynamic": "float32", "materialized": "float32"},
    }


def _fixture() -> dict[str, object]:
    return {
        "row_id": "fixture-0",
        "prompt_ids_sha256": "prompt",
        "processor_fingerprint": "processor",
        "image_sha256": "image",
        "generation_fingerprint": "generation",
    }


def test_parity_receipt_binds_execution_identity_and_qualifies_receipt(
    tmp_path: Path,
) -> None:
    execution_model = resolve_execution_model(
        base_model_path=_write_snapshot(tmp_path / "base"),
        target_dtype="bf16",
    )
    parity = build_execution_model_parity_receipt(
        execution_model=execution_model,
        fixture_identity=_fixture(),
        comparison=_comparison(),
    )
    assert validate_execution_model_parity_receipt(
        parity,
        execution_model=execution_model,
    ) == parity

    qualified = bind_execution_model_parity(execution_model, parity)
    assert qualified["qualification"]["parity_digest"] == parity["digest"]
    assert validate_execution_model_receipt(qualified) == qualified


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
        (("comparison", "exact_checks", "prompt_ids"), False),
        (("comparison", "selected_vocab", "allclose"), False),
        (("thresholds", "selected_vocab", "atol"), 1.0),
    ],
)
def test_parity_receipt_mutation_fails(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    execution_model = resolve_execution_model(
        base_model_path=_write_snapshot(tmp_path / "base"),
        target_dtype="bf16",
    )
    parity = build_execution_model_parity_receipt(
        execution_model=execution_model,
        fixture_identity=_fixture(),
        comparison=_comparison(),
    )
    mutated = copy.deepcopy(parity)
    owner = mutated
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value

    with pytest.raises(RuntimeContractError):
        validate_execution_model_parity_receipt(
            mutated,
            execution_model=execution_model,
        )
