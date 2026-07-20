from __future__ import annotations

import pytest
import json
from types import SimpleNamespace

from src.config.fingerprint import sha256_json
from src.inference.backend import token_ids_sha256
from scripts.research.build_exact_trace_coordinate_state_bank import (
    AssemblyError,
    _coord_offsets,
    _load_frozen_trace_identity,
    _load_prompt_contract,
    _pad_interval,
    _validate_source_trace_binding,
    _validate_trace_prompt_identity,
)


def test_coordinate_offsets_use_exact_discrete_token_values() -> None:
    tokens = [151646, 151648, 151670, 151802, 152230, 152669, 151649]
    assert _coord_offsets(tokens, case_id="unit", expected_bins=[0, 132, 560, 999]) == [2, 3, 4, 5]


def test_coordinate_offsets_reject_a_wrong_exact_token() -> None:
    tokens = [151646, 151670, 151802, 152231, 152669, 151649]
    with pytest.raises(AssemblyError, match="parsed coordinate bins disagree"):
        _coord_offsets(tokens, case_id="unit", expected_bins=[0, 132, 560, 999])


def test_image_pad_interval_requires_one_contiguous_run() -> None:
    assert _pad_interval([1, 7, 7, 7, 2], image_pad_token_id=7, case_id="unit") == (1, 4)
    with pytest.raises(AssemblyError, match="one contiguous image-pad run"):
        _pad_interval([7, 1, 7], image_pad_token_id=7, case_id="unit")


def test_trace_prompt_hash_is_checked_separately_from_bank_prompt_contract() -> None:
    prompt_ids = [11, 12, 13]
    prompt_hash = token_ids_sha256(prompt_ids)
    identity = {
        "passed": True,
        "checks": {
            "chat_text_sha256": True,
            "executed_media_sha256": True,
            "executed_prompt_token_ids_sha256": True,
            "height": True,
            "image_sha256": True,
            "observed_image_grid_thw": True,
            "prompt_token_ids_sha256": True,
            "width": True,
        },
        "expected": {
            "chat_text_sha256": "a" * 64,
            "executed_media_sha256": "b" * 64,
            "executed_prompt_token_ids_sha256": prompt_hash,
            "height": 8,
            "image_sha256": "c" * 64,
            "observed_image_grid_thw": [1, 1, 1],
            "prompt_token_ids_sha256": prompt_hash,
            "width": 12,
        },
    }
    identity["observed"] = dict(identity["expected"])
    image = {"identity_check": identity}
    prompt = {
        "chat_text_sha256": "a" * 64,
        "prompt_token_ids_sha256": prompt_hash,
        "image_sha256": "c" * 64,
        "width": 12,
        "height": 8,
    }

    # The bank-level prompt contract is intentionally not passed here: the
    # trace hash is an executed image-prompt identity and has a different role.
    _validate_trace_prompt_identity(image, prompt, prompt_ids, case_id="unit")


def test_trace_prompt_identity_rejects_failed_identity_check() -> None:
    prompt_ids = [11]
    prompt_hash = token_ids_sha256(prompt_ids)
    with pytest.raises(AssemblyError, match="identity_check did not pass"):
        _validate_trace_prompt_identity(
            {"identity_check": {"passed": False}},
            {"prompt_token_ids_sha256": prompt_hash},
            prompt_ids,
            case_id="unit",
        )


def _write_trusted_trace(path) -> tuple[str, str]:
    prompt_ids = [11, 12, 13]
    prompt_hash = token_ids_sha256(prompt_ids)
    chat_hash = "a" * 64
    image_hash = "b" * 64
    identity = {
        "passed": True,
        "checks": {
            "chat_text_sha256": True,
            "executed_media_sha256": True,
            "executed_prompt_token_ids_sha256": True,
            "height": True,
            "image_sha256": True,
            "observed_image_grid_thw": True,
            "prompt_token_ids_sha256": True,
            "width": True,
        },
        "expected": {
            "chat_text_sha256": chat_hash,
            "executed_media_sha256": "c" * 64,
            "executed_prompt_token_ids_sha256": prompt_hash,
            "height": 8,
            "image_sha256": image_hash,
            "observed_image_grid_thw": [1, 1, 1],
            "prompt_token_ids_sha256": prompt_hash,
            "width": 12,
        },
    }
    identity["observed"] = dict(identity["expected"])
    infer_config_path = path.with_name("trusted-infer.yaml")
    infer_config_path.write_text("schema_version: 1\n", encoding="utf-8")
    payload = {
        "config": {
            "infer_config": str(infer_config_path),
            "resolved_config_fingerprint": "d" * 64,
        },
        "images": [
            {
                "image_id": "1",
                "prompt": {
                    "chat_text_sha256": chat_hash,
                    "height": 8,
                    "image_sha256": image_hash,
                    "prompt_token_ids": prompt_ids,
                    "prompt_token_ids_sha256": prompt_hash,
                    "width": 12,
                },
                "identity_check": identity,
                "extended_root_greedy": {"rows": []},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return "d" * 64, chat_hash


def test_prompt_contract_is_bound_to_bank_identity(tmp_path) -> None:
    contract = {"assistant_format": "object_box_closed", "user": "same"}
    contract_path = tmp_path / "prompt-contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": "coordexp.rollout_calibration.prompt_contract.v1",
                "contract": contract,
                "prompt_identity_sha256": sha256_json(contract),
            }
        ),
        encoding="utf-8",
    )
    identity = _load_prompt_contract(
        contract_path,
        expected_prompt_identity_sha256=sha256_json(contract),
    )
    assert identity["sha256"]
    bad_path = tmp_path / "bad-contract.json"
    bad_path.write_text(contract_path.read_text(encoding="utf-8").replace("same", "foreign"), encoding="utf-8")
    with pytest.raises(AssemblyError, match="prompt contract identity mismatch"):
        _load_prompt_contract(
            bad_path,
            expected_prompt_identity_sha256=identity["prompt_identity_sha256"],
        )


def test_source_trace_must_match_frozen_config_and_chat_identity(
    tmp_path, monkeypatch
) -> None:
    trusted_path = tmp_path / "trusted-trace.json"
    config_hash, chat_hash = _write_trusted_trace(trusted_path)
    contract = {
        "assistant_format": "object_box_closed",
        "object_field_order": "desc_first",
        "object_ordering": "geo_sorted",
        "system": "system",
        "user": "user",
    }
    resolved = SimpleNamespace(
        fingerprint=config_hash,
        config=SimpleNamespace(
            template=SimpleNamespace(
                assistant_format=contract["assistant_format"],
                object_field_order=contract["object_field_order"],
                object_ordering=contract["object_ordering"],
                prompt=SimpleNamespace(
                    system=contract["system"], user=contract["user"]
                ),
            )
        ),
    )
    monkeypatch.setattr(
        "scripts.research.build_exact_trace_coordinate_state_bank.load_infer_config",
        lambda _: resolved,
    )
    frozen = _load_frozen_trace_identity(
        trusted_path, canonical_prompt_contract=contract
    )
    assert frozen["resolved_config_fingerprint"] == config_hash
    assert frozen["chat_text_sha256"] == chat_hash

    observed = _validate_source_trace_binding(
        {"config": {"resolved_config_fingerprint": config_hash}},
        {"prompt": {"chat_text_sha256": chat_hash}},
        frozen,
        case_id="candidate",
    )
    assert observed == {
        "resolved_config_fingerprint": config_hash,
        "chat_text_sha256": chat_hash,
    }
    with pytest.raises(AssemblyError, match="canonical chat-text hash differs"):
        _validate_source_trace_binding(
            {"config": {"resolved_config_fingerprint": config_hash}},
            {"prompt": {"chat_text_sha256": "e" * 64}},
            frozen,
            case_id="foreign-prompt",
        )
    with pytest.raises(AssemblyError, match="resolved config fingerprint differs"):
        _validate_source_trace_binding(
            {"config": {"resolved_config_fingerprint": "f" * 64}},
            {"prompt": {"chat_text_sha256": chat_hash}},
            frozen,
            case_id="foreign-config",
        )

    foreign_contract = {**contract, "user": "foreign user"}
    with pytest.raises(
        AssemblyError,
        match="does not instantiate the canonical prompt contract",
    ):
        _load_frozen_trace_identity(
            trusted_path, canonical_prompt_contract=foreign_contract
        )
