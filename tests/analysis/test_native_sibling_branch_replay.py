from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.research.run_native_sibling_branch_replay import (
    _default_seed,
    _build_reconstructed_prompt_record,
    _parse_result_receipt,
    _sha256_json,
    _seed_schedule,
    check_shard_receipts,
    extract_branch_row_tokens,
    _execution_contract,
    merge_shard_receipts,
    reconstruct_branch_prompt,
    reconstruct_donor_prefix,
    select_runtime_prompt_token_ids,
    validate_donor_lineage,
    validate_cli_args,
)


OBJECT_REF_START = 151646
BOX_END = 151649


def _row(description: int, x1: int) -> list[int]:
    return [OBJECT_REF_START, description, 151647, 151648, 151670 + x1, 151671, 151672, 151673, BOX_END]


def _parser_row(description: str = "person") -> str:
    return (
        f"<|object_ref_start|>{description}<|object_ref_end|>"
        "<|box_start|><|coord_10|><|coord_20|><|coord_30|><|coord_40|>"
        "<|box_end|>"
    )


def _parse_probe_result(*, parser_text: str, generated_token_ids: list[int], stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="horizon-probe",
        parser_text=parser_text,
        generated_token_ids=generated_token_ids,
        stop_reason=stop_reason,
    )


def _parse_probe_raw() -> SimpleNamespace:
    return SimpleNamespace(image=SimpleNamespace(width=864, height=1152))


def _donor() -> dict[str, object]:
    first = _row(100, 1)
    second = _row(200, 2)
    third = _row(300, 3)
    return {
        "request_id": "donor",
        "execution_evidence": {"image_id": "12576"},
        "decode_result": {
            "prompt_token_ids": [11, 12, 13],
            "generated_token_ids": [*first, *second, *third, 151645],
        },
    }


def _current_call_bundle() -> dict[str, object]:
    prompt = [11, 12, 13, *_row(100, 1)]
    model_identity = {"family": "model"}
    tokenizer_identity = {"family": "tokenizer"}
    generation_fingerprint = "g" * 64
    return {
        "schema_version": "native_sibling_branch_replay.call_bundle.v1",
        "image_id": "12576",
        "donor": {
            "source_image_sha256": "a" * 64,
            "source_image_width": 864,
            "source_image_height": 1152,
        },
        "runtime": {
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "generation_config_fingerprint": generation_fingerprint,
            "attention_implementation": "sdpa",
        },
        "prompt": {
            "prompt_token_count": len(prompt),
            "prompt_token_ids_sha256": _sha256_json(prompt),
        },
        "decode_result": {
            "prompt_token_ids": prompt,
            "generated_token_ids": _row(200, 2),
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "generation_config_fingerprint": generation_fingerprint,
            "execution_receipt": {
                "attention_implementation": "sdpa",
                "schema_version": "decode_execution_receipt.v1",
            },
        },
    }


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "mode": "run",
        "image_id": "12576",
        "output_root": Path("/tmp/native-sibling-test"),
        "infer_config": Path("config.yaml"),
        "source_jsonl": Path("source.jsonl"),
        "sampled_runtime_attestation": Path("attestation.json"),
        "donor_bundle": Path("donor.json"),
        "donor_prefix_token_count": 9,
        "branch_row_span": None,
        "branch_row_index": None,
        "branch_row_bundle": None,
        "branch_label": "test",
        "seed": None,
        "seeds": None,
        "num_seeds": 32,
        "seed_root": 2026071301,
        "shard_index": 0,
        "shard_count": 1,
        "max_new_tokens": 512,
        "temperature": 0.4,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "runtime_dtype": "config",
        "no_greedy_control": False,
        "exact_donor_token_prompt": False,
        "use_local_sampling_context": False,
        "shard_receipt": None,
        "merged_output": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _receipt(path: Path, *, request_id: str, seed: int | None, calls: list[dict[str, object]] | None = None) -> None:
    payload = {
        "schema_version": "native_sibling_branch_replay.shard_receipt.v1",
        "execution_contract": {
            "physical_batch_size": 1,
            "runtime_dtype_mode": "config",
            "model_config_dtype": "bf16",
            "attention_implementation": "sdpa",
            "decode_generation_policy": {
                "mode": "sampled",
                "temperature": 0.4,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
            },
        },
        "calls": calls
        or [{"request_id": request_id, "sampling_seed": seed, "bundle_path": "call.json"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_exact_prefix_reconstruction_uses_only_donor_tokens() -> None:
    donor = _donor()
    prefix = reconstruct_donor_prefix(donor, 9)
    assert prefix["donor_prefix_token_ids"] == _row(100, 1)
    assert prefix["reconstructed_prompt_token_ids"] == [11, 12, 13, *_row(100, 1)]
    branch = extract_branch_row_tokens(donor, row_index=1, prefix_token_count=9)
    assert branch is not None
    assert branch["token_ids"] == _row(200, 2)
    rebuilt = reconstruct_branch_prompt(
        donor,
        prefix_token_count=9,
        branch_row_index=1,
    )
    assert rebuilt["assistant_continuation_token_ids"] == [*_row(100, 1), *_row(200, 2)]
    assert rebuilt["expected_prompt_token_ids"] == [11, 12, 13, *_row(100, 1), *_row(200, 2)]


def test_row_zero_prompt_uses_ordinary_prompt_without_empty_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_prompt_record(raw: object, template_config: object, **kwargs: object) -> str:
        captured.update(kwargs)
        return "ordinary-prompt"

    monkeypatch.setattr("src.inference.prompt.build_prompt_record", fake_build_prompt_record)

    class NoDecodeTokenizer:
        def decode(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("row-zero path must not decode an empty continuation")

    result = _build_reconstructed_prompt_record(
        "raw",
        "template",
        processor="processor",
        row_index=0,
        tokenizer=NoDecodeTokenizer(),
        continuation_token_ids=[],
    )
    assert result == "ordinary-prompt"
    assert captured == {"processor": "processor", "row_index": 0}


def test_exact_donor_token_prompt_extends_active_base_without_retokenizing() -> None:
    row = _row(100, 1)
    selected, mode = select_runtime_prompt_token_ids(
        built_prompt_token_ids=[11, 12, 13],
        expected_prompt_token_ids=[11, 12, 13, *row],
        exact_donor_token_prompt=True,
    )
    assert selected == [11, 12, 13, *row]
    assert mode == "exact_donor_token_ids"

    with pytest.raises(RuntimeError, match="does not extend the active base prompt"):
        select_runtime_prompt_token_ids(
            built_prompt_token_ids=[11, 99, 13],
            expected_prompt_token_ids=[11, 12, 13, *row],
            exact_donor_token_prompt=True,
        )


def test_distinct_native_child_bundle_uses_relative_generated_row_coordinates() -> None:
    donor = _donor()
    parent_prefix = [*_row(100, 1)]
    child = {
        "decode_result": {
            "prompt_token_ids": [11, 12, 13, *parent_prefix],
            "generated_token_ids": _row(200, 2),
        }
    }
    rebuilt = reconstruct_branch_prompt(
        donor,
        prefix_token_count=len(parent_prefix),
        branch_bundle=child,
        branch_row_span=(0, 9),
    )
    assert rebuilt["branch"]["token_ids"] == _row(200, 2)
    mismatched = {
        "decode_result": {
            "prompt_token_ids": [11, 12, 99, *parent_prefix],
            "generated_token_ids": _row(200, 2),
        }
    }
    with pytest.raises(ValueError, match="distinct branch bundle prompt"):
        reconstruct_branch_prompt(
            donor,
            prefix_token_count=len(parent_prefix),
            branch_bundle=mismatched,
            branch_row_span=(0, 9),
        )


def test_current_call_bundle_round_trips_as_a_strict_branch_source() -> None:
    donor = _donor()
    child = _current_call_bundle()
    rebuilt = reconstruct_branch_prompt(
        donor,
        prefix_token_count=9,
        branch_bundle=child,
        branch_row_span=(0, 9),
    )
    assert rebuilt["branch"]["token_ids"] == _row(200, 2)
    lineage = validate_donor_lineage(
        child,
        image_id="12576",
        source_image_sha256="a" * 64,
        source_width=864,
        source_height=1152,
        model_identity={"family": "model"},
        tokenizer_identity={"family": "tokenizer"},
        generation_config_fingerprint="g" * 64,
        attention_implementation="sdpa",
        strict_runtime=True,
    )
    assert lineage["image_id"] == "12576"
    assert lineage["source_image_sha256"] == "a" * 64


def test_original_donor_row_must_start_exactly_at_parent_prefix() -> None:
    with pytest.raises(ValueError, match="exactly at the parent prefix"):
        extract_branch_row_tokens(_donor(), row_index=2, prefix_token_count=9)


def test_default_seed_schedule_pairs_sibling_labels_and_changes_parent_state() -> None:
    args = _args(seeds=None, num_seeds=4, shard_count=1, shard_index=0)
    first = _seed_schedule(
        args,
        image_id="12576",
        parent_prompt_hash="a" * 64,
        parent_prefix_hash="b" * 64,
        branch_label="pizza",
    )
    second = _seed_schedule(
        args,
        image_id="12576",
        parent_prompt_hash="a" * 64,
        parent_prefix_hash="b" * 64,
        branch_label="cup",
    )
    changed = _seed_schedule(
        args,
        image_id="12576",
        parent_prompt_hash="c" * 64,
        parent_prefix_hash="b" * 64,
        branch_label="pizza",
    )
    assert first == second
    assert first != changed
    assert first == [
        _default_seed(
            seed_root=2026071301,
            image_id="12576",
            parent_prompt_hash="a" * 64,
            parent_prefix_hash="b" * 64,
            index=index,
        )
        for index in range(4)
    ]


def test_donor_lineage_rejects_image_and_runtime_mismatch() -> None:
    donor = _donor()
    donor["execution_evidence"] = {
        "image_id": "12576",
        "source_image_sha256": "a" * 64,
        "source_width": 864,
        "source_height": 1152,
    }
    donor["decode_result"]["model_identity"] = {"family": "model"}
    donor["decode_result"]["tokenizer_identity"] = {"family": "tokenizer"}
    donor["decode_result"]["generation_config_fingerprint"] = "g" * 64
    donor["decode_result"]["execution_receipt"] = {
        "model_identity_fingerprint": "m" * 64,
        "tokenizer_identity_fingerprint": "t" * 64,
        "attention_implementation": "sdpa",
        "schema_version": "decode_execution_receipt.v1",
    }
    with pytest.raises(ValueError, match="donor image id"):
        validate_donor_lineage(donor, image_id="19432")
    with pytest.raises(ValueError, match="donor source image digest"):
        validate_donor_lineage(donor, image_id="12576", source_image_sha256="b" * 64)
    with pytest.raises(ValueError, match="donor model identity"):
        validate_donor_lineage(
            donor,
            image_id="12576",
            model_identity={"family": "other-model"},
            strict_runtime=True,
        )


def test_horizon_projection_requires_natural_stop_for_terminal_classification() -> None:
    natural = _parse_result_receipt(
        _parse_probe_result(
            parser_text=_parser_row() + "<|im_end|>",
            generated_token_ids=_row(100, 1),
            stop_reason="im_end",
        ),
        _parse_probe_raw(),
    )["horizon_projection"]
    assert natural["termination_classification"] == "natural_termination_before_horizon"
    assert natural["natural_termination_before_horizon"] is True
    assert natural["parser_valid"] is True

    token_limited = _parse_result_receipt(
        _parse_probe_result(
            parser_text=_parser_row(),
            generated_token_ids=_row(100, 1),
            stop_reason="length",
        ),
        _parse_probe_raw(),
    )["horizon_projection"]
    assert token_limited["termination_classification"] == "token_limit_truncated"
    assert token_limited["natural_termination_before_horizon"] is False

    malformed = _parse_result_receipt(
        _parse_probe_result(
            parser_text="<|object_ref_start|>unterminated",
            generated_token_ids=[],
            stop_reason="im_end",
        ),
        _parse_probe_raw(),
    )["horizon_projection"]
    assert malformed["termination_classification"] == "invalid_or_malformed"
    assert malformed["natural_termination_before_horizon"] is False
    assert malformed["parser_valid"] is False


def test_horizon_projection_retains_four_row_projection() -> None:
    generated = [token for index in range(4) for token in _row(100 + index, index + 1)]
    parser_text = "".join(_parser_row(str(100 + index)) for index in range(4)) + "<|im_end|>"
    projection = _parse_result_receipt(
        _parse_probe_result(
            parser_text=parser_text,
            generated_token_ids=generated,
            stop_reason="im_end",
        ),
        _parse_probe_raw(),
    )["horizon_projection"]
    assert projection["termination_classification"] == "horizon_reached"
    assert projection["natural_termination_before_horizon"] is False
    assert len(projection["first_four_complete_row_spans"]) == 4


def test_prefix_must_end_at_complete_row_boundary() -> None:
    with pytest.raises(ValueError, match="complete row boundary"):
        reconstruct_donor_prefix(_donor(), 2)


def test_cli_rejects_non_single_physical_batch_or_wrong_temperature() -> None:
    args = _args()
    validate_cli_args(args)
    with pytest.raises(SystemExit, match="temperature"):
        validate_cli_args(_args(temperature=0.2))
    with pytest.raises(SystemExit, match="seed values must be unique"):
        validate_cli_args(_args(seeds=[4, 4]))


def test_per_call_policy_receipt_is_not_silently_greedy_sampled() -> None:
    sampled = {"mode": "sampled", "temperature": 0.4, "top_p": 0.95, "repetition_penalty": 1.0}
    greedy = {"mode": "greedy", "temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0}
    common = {
        "model_identity": {"family": "test"},
        "tokenizer_identity": {"family": "test"},
        "generation_fingerprint": "g" * 64,
        "config_dtype": "bf16",
        "runtime_dtype": "config",
        "dtype_summary": {"parameter_dtype_names": ["torch.bfloat16"]},
        "attention_implementation": "sdpa",
        "sampling_attestation_mode": "persisted-bf16-capability",
    }
    sampled_contract = _execution_contract(policy=sampled, **common)
    greedy_contract = _execution_contract(
        policy=greedy,
        **{**common, "sampling_attestation_mode": "not_applicable_greedy"},
    )
    assert sampled_contract["decode_generation_policy"]["mode"] == "sampled"
    assert greedy_contract["decode_generation_policy"]["mode"] == "greedy"
    assert greedy_contract["sampling_attestation_mode"] == "not_applicable_greedy"


def test_merge_rejects_request_and_seed_collisions(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _receipt(first, request_id="a", seed=10)
    _receipt(second, request_id="b", seed=10)
    with pytest.raises(ValueError, match="duplicate sampling_seed"):
        merge_shard_receipts([first, second])
    _receipt(second, request_id="a", seed=11)
    with pytest.raises(ValueError, match="duplicate request_id"):
        merge_shard_receipts([first, second])


def test_merge_is_deterministic_and_checks_batch_one(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _receipt(first, request_id="b", seed=20)
    _receipt(second, request_id="a", seed=10)
    left = merge_shard_receipts([first, second])
    right = merge_shard_receipts([second, first])
    assert left == right
    check = check_shard_receipts([first, second])
    assert check["call_count"] == 2
    assert check["stable_attribution_check"]["physical_batch_size_all_one"] is True
