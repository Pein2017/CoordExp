from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import scripts.research.collect_vllm_trajectory_panel as collector
from scripts.research.collect_vllm_trajectory_panel import (
    SCHEMA_VERSION,
    SOURCE_B16_ROW_BUDGET,
    _decode_modes,
    _materialize_completion,
    _ordered_completions,
    _parse_args,
    _generation_health,
    _source_b16_receipt,
    _source_b16_summary,
    _stable_session_identity,
    _validated_resume_batch,
    build_artifact,
    parse_image_ids,
    resolve_panel_execution_model,
    sampling_params_kwargs,
    shard_examples,
)
from src.inference.parsing import parse_compact_object_box_closed
from src.templates.renderer import BOX_END_TOKEN


def test_image_sharding_is_disjoint_stable_and_complete() -> None:
    rows = list(range(17))
    shards = [shard_examples(rows, worker_index=index, worker_count=8) for index in range(8)]
    assert shards[0] == [0, 8, 16]
    assert shards[7] == [7, 15]
    assert sorted(value for shard in shards for value in shard) == rows
    with pytest.raises(ValueError, match="worker_index"):
        shard_examples(rows, worker_index=8, worker_count=8)
    assert parse_image_ids("941, 1083") == {"941", "1083"}
    with pytest.raises(ValueError, match="at least one"):
        parse_image_ids(" , ")


def test_sampling_policy_is_one_greedy_and_sixteen_samples() -> None:
    greedy = sampling_params_kwargs(
        decode_mode="greedy",
        sample_count=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=512,
        stop_token_id=99,
        seed=None,
    )
    assert greedy["n"] == 1
    assert greedy["temperature"] == 0.0
    assert "seed" not in greedy

    source_b16 = sampling_params_kwargs(
        decode_mode="source_b16",
        sample_count=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=2048,
        stop_token_id=99,
        seed=None,
    )
    assert source_b16 == {
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_tokens": 2048,
        "stop_token_ids": [99],
        "ignore_eos": False,
        "detokenize": True,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": True,
    }
    source_b16_rp1p10 = sampling_params_kwargs(
        decode_mode="source_b16",
        sample_count=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.1,
        max_new_tokens=2048,
        stop_token_id=99,
        seed=None,
    )
    assert source_b16_rp1p10["repetition_penalty"] == 1.1
    with pytest.raises(ValueError, match="exactly 1.0 or 1.1"):
        sampling_params_kwargs(
            decode_mode="source_b16",
            sample_count=16,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.2,
            max_new_tokens=2048,
            stop_token_id=99,
            seed=None,
        )
    with pytest.raises(ValueError, match="require repetition_penalty=1.0"):
        sampling_params_kwargs(
            decode_mode="sampled",
            sample_count=16,
            temperature=0.4,
            top_p=0.95,
            repetition_penalty=1.1,
            max_new_tokens=512,
            stop_token_id=99,
            seed=31_001,
        )

    sampled = sampling_params_kwargs(
        decode_mode="sampled",
        sample_count=16,
        temperature=0.4,
        top_p=0.95,
        repetition_penalty=1.0,
        max_new_tokens=512,
        stop_token_id=99,
        seed=31_001,
    )
    assert sampled["n"] == 16
    assert sampled["temperature"] == 0.4
    assert sampled["top_p"] == 0.95
    assert sampled["repetition_penalty"] == 1.0
    assert sampled["seed"] == 31_001


def test_sampled_only_panel_uses_no_greedy_artifact_mode() -> None:
    assert _decode_modes(sampled_only=False) == ("greedy", "sampled")
    assert _decode_modes(sampled_only=True) == ("sampled",)
    assert _decode_modes(sampled_only=False, source_b16=True) == ("source_b16",)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _decode_modes(sampled_only=True, source_b16=True)


def test_source_b16_cli_is_mutually_exclusive_with_sampled_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = [
        "collect_vllm_trajectory_panel.py",
        "--output-root",
        "/tmp/panel",
        "--worker-index",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", [*base, "--source-b16"])
    assert _parse_args().source_b16 is True
    assert _parse_args().source_b16_repetition_penalty == 1.0
    monkeypatch.setattr(
        sys,
        "argv",
        [*base, "--source-b16", "--source-b16-repetition-penalty", "1.1"],
    )
    assert _parse_args().source_b16_repetition_penalty == 1.1
    monkeypatch.setattr(sys, "argv", [*base, "--source-b16", "--sampled-only"])
    with pytest.raises(SystemExit):
        _parse_args()


@dataclass
class Completion:
    index: int
    token_ids: list[int]
    finish_reason: str


class Tokenizer:
    def decode(self, values: list[int], **_: object) -> str:
        return ",".join(str(value) for value in values)


class SourceTokenizer:
    box_end_token_id = 900
    im_end_token_id = 901

    def __init__(self, pieces: dict[int, str]) -> None:
        self._pieces = {
            self.box_end_token_id: BOX_END_TOKEN,
            self.im_end_token_id: "<|im_end|>",
            **pieces,
        }

    def encode(self, value: str, **_: object) -> list[int]:
        assert value == BOX_END_TOKEN
        return [self.box_end_token_id]

    def decode(self, values: list[int], **_: object) -> str:
        return "".join(self._pieces[value] for value in values)


def _valid_source_row(name: str) -> str:
    return (
        f"<|object_ref_start|>{name}<|object_ref_end|><|box_start|>"
        "<|coord_10|><|coord_10|><|coord_20|><|coord_20|>"
    )


def _source_b16_receipt_for(
    token_ids: list[int],
    tokenizer: SourceTokenizer,
    *,
    stop_reason: str,
    row_id: str = "example-1:source-b16",
) -> tuple[str, dict[str, object]]:
    parser_ids = token_ids[:-1] if stop_reason == "im_end" else token_ids
    text = tokenizer.decode(parser_ids)
    raw_parser = parse_compact_object_box_closed(
        text,
        row_id=row_id,
        row_index=0,
        image_width=100,
        image_height=100,
    )
    receipt = _source_b16_receipt(
        token_ids=token_ids,
        parser_text=text,
        stop_reason=stop_reason,
        tokenizer=tokenizer,
        stop_token_id=tokenizer.im_end_token_id,
        row_id=row_id,
        image_width=100,
        image_height=100,
        raw_parser=raw_parser,
    )
    return text, receipt


def _source_b16_rollout(
    *,
    image_id: int,
    example_id: str,
    token_ids: list[int],
    tokenizer: SourceTokenizer,
    stop_reason: str,
) -> dict[str, object]:
    row_id = f"{example_id}:source-b16"
    text, receipt = _source_b16_receipt_for(
        token_ids,
        tokenizer,
        stop_reason=stop_reason,
        row_id=row_id,
    )
    raw_parser = parse_compact_object_box_closed(
        text,
        row_id=row_id,
        row_index=0,
        image_width=100,
        image_height=100,
    )
    return {
        "image_id": image_id,
        "example_id": example_id,
        "trajectory_id": "source-b16",
        "decode_mode": "source_b16",
        "generated_token_ids": token_ids,
        "generated_token_ids_sha256": hashlib.sha256(
            json.dumps(
                token_ids,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest(),
        "generated_text": text,
        "stop_reason": stop_reason,
        "image_width": 100,
        "image_height": 100,
        "prompt_token_ids": [1],
        "prompt_token_ids_sha256": "prompt-hash",
        "source_image_file_sha256": f"{image_id:064x}",
        "executed_rgb_sha256": f"{image_id + 100:064x}",
        "predictions": raw_parser.to_artifact_dict(),
        "source_b16": receipt,
    }


def test_completion_order_and_terminal_materialization() -> None:
    native = type(
        "Native",
        (),
        {"outputs": [Completion(1, [7, 99], "stop"), Completion(0, [6], "length")]},
    )()
    ordered = _ordered_completions(native, expected_count=2)
    assert [item.index for item in ordered] == [0, 1]
    assert _materialize_completion(
        completion=ordered[0], tokenizer=Tokenizer(), stop_token_id=99
    ) == ([6], "6", "length")
    assert _materialize_completion(
        completion=ordered[1], tokenizer=Tokenizer(), stop_token_id=99
    ) == ([7, 99], "7", "im_end")


def test_source_b16_projects_first_sixteen_valid_rows_and_keeps_raw_repeats() -> None:
    pieces = {index: _valid_source_row(f"row-{index}") for index in range(17)}
    tokenizer = SourceTokenizer(pieces)
    token_ids = [token for index in range(17) for token in (index, tokenizer.box_end_token_id)]
    _, receipt = _source_b16_receipt_for(token_ids, tokenizer, stop_reason="length")

    assert receipt["status"] == "accepted_budget"
    assert receipt["raw_valid_complete_row_count"] == 17
    assert receipt["projected_valid_complete_row_count"] == SOURCE_B16_ROW_BUDGET
    assert receipt["projected_token_end_offset_exclusive"] == SOURCE_B16_ROW_BUDGET * 2
    assert receipt["projected_token_ids"] == token_ids[: SOURCE_B16_ROW_BUDGET * 2]
    assert "row-15" in receipt["projected_text"]
    assert "row-16" not in receipt["projected_text"]
    assert receipt["token_limit_before_budget"] is False
    assert receipt["natural_end_before_budget"] is False
    assert receipt["projected_parser_evidence"]["parse_status"] == "accepted"


def test_source_b16_accepts_clean_natural_end_before_budget_and_strips_terminal_im_end() -> None:
    pieces = {index: _valid_source_row(f"row-{index}") for index in range(3)}
    tokenizer = SourceTokenizer(pieces)
    token_ids = [
        token
        for index in range(3)
        for token in (index, tokenizer.box_end_token_id)
    ] + [tokenizer.im_end_token_id]
    text, receipt = _source_b16_receipt_for(token_ids, tokenizer, stop_reason="im_end")

    assert receipt["status"] == "accepted_natural_end"
    assert receipt["raw_valid_complete_row_count"] == 3
    assert receipt["projected_valid_complete_row_count"] == 3
    assert receipt["projected_token_end_offset_exclusive"] == len(token_ids) - 1
    assert receipt["projected_token_ids"] == token_ids[:-1]
    assert receipt["projected_text"] == text
    assert "<|im_end|>" not in receipt["projected_text"]
    assert receipt["token_limit_before_budget"] is False
    assert receipt["natural_end_before_budget"] is True


def test_source_b16_rejects_length_before_budget_but_accepts_length_after_budget() -> None:
    short_tokenizer = SourceTokenizer(
        {index: _valid_source_row(f"row-{index}") for index in range(15)}
    )
    short_ids = [
        token
        for index in range(15)
        for token in (index, short_tokenizer.box_end_token_id)
    ]
    _, short_receipt = _source_b16_receipt_for(short_ids, short_tokenizer, stop_reason="length")
    assert short_receipt["status"] == "failed_token_limit_before_budget"
    assert short_receipt["projected_valid_complete_row_count"] == 15
    assert short_receipt["token_limit_before_budget"] is True

    complete_tokenizer = SourceTokenizer(
        {index: _valid_source_row(f"row-{index}") for index in range(16)}
    )
    complete_ids = [
        token
        for index in range(16)
        for token in (index, complete_tokenizer.box_end_token_id)
    ]
    _, complete_receipt = _source_b16_receipt_for(
        complete_ids, complete_tokenizer, stop_reason="length"
    )
    assert complete_receipt["status"] == "accepted_budget"
    assert complete_receipt["token_limit_before_budget"] is False


def test_source_b16_fails_closed_before_budget_but_ignores_malformed_tail_after_budget() -> None:
    malformed_before = "<|object_ref_start|>broken<|object_ref_end|><|box_start|><|coord_10|>"
    before_pieces = {0: malformed_before}
    before_pieces.update({index: _valid_source_row(f"row-{index}") for index in range(1, 17)})
    before_tokenizer = SourceTokenizer(before_pieces)
    before_ids = [
        token
        for index in range(17)
        for token in (index, before_tokenizer.box_end_token_id)
    ]
    _, before_receipt = _source_b16_receipt_for(
        before_ids, before_tokenizer, stop_reason="length"
    )
    assert before_receipt["status"] == "failed_invalid_before_budget"
    assert before_receipt["projected_valid_complete_row_count"] == 0
    assert before_receipt["failure_parser_evidence"]["dropped_prediction_count"] == 1

    malformed_after = "<|object_ref_start|>broken<|object_ref_end|><|box_start|><|coord_10|>"
    after_pieces = {index: _valid_source_row(f"row-{index}") for index in range(16)}
    after_pieces[16] = malformed_after
    after_tokenizer = SourceTokenizer(after_pieces)
    after_ids = [
        token
        for index in range(17)
        for token in (index, after_tokenizer.box_end_token_id)
    ]
    _, after_receipt = _source_b16_receipt_for(after_ids, after_tokenizer, stop_reason="length")
    assert after_receipt["status"] == "accepted_budget"
    assert after_receipt["projected_valid_complete_row_count"] == SOURCE_B16_ROW_BUDGET
    assert after_receipt["raw_parser_evidence"]["parse_status"] == "accepted_with_drops"


def test_source_b16_fails_closed_for_invalid_geometry_before_budget() -> None:
    invalid_row = (
        "<|object_ref_start|>broken<|object_ref_end|><|box_start|>"
        "<|coord_20|><|coord_20|><|coord_10|><|coord_10|>"
    )
    tokenizer = SourceTokenizer({0: invalid_row})
    token_ids = [0, tokenizer.box_end_token_id, tokenizer.im_end_token_id]
    _, receipt = _source_b16_receipt_for(token_ids, tokenizer, stop_reason="im_end")

    assert receipt["status"] == "failed_invalid_before_budget"
    assert receipt["failure_parser_evidence"]["dropped_predictions"][0]["reason"] == "geometry_invalid"


def test_generation_health_is_computed_from_materialized_rows() -> None:
    health = _generation_health(
        [
            {
                "generated_token_ids": [1, 2],
                "stop_reason": "im_end",
                "predictions": {"parse_status": "accepted"},
            },
            {
                "generated_token_ids": [3],
                "stop_reason": "length",
                "predictions": {"parse_status": "accepted_with_drops"},
            },
        ],
        elapsed_seconds=2.0,
    )
    assert health["generated_token_count"] == 3
    assert health["stop_reason_counts"] == {"im_end": 1, "length": 1}
    assert health["natural_closure_count"] == 1
    assert health["routes_per_second"] == 1.0


def test_stable_session_identity_excludes_rank_local_runtime_observations() -> None:
    class Receipt:
        def to_artifact_dict(self) -> dict[str, object]:
            return {
                "backend": "vllm",
                "effective_settings": {
                    "batch_size": 32,
                    "engine_kwargs": {"max_num_seqs": 32},
                    "runtime_preflight": {"process": {"pid": 123}},
                    "performance": {"request_count": 8},
                },
            }

    identity = _stable_session_identity(Receipt())
    assert identity["effective_settings"] == {
        "batch_size": 32,
        "engine_kwargs": {"max_num_seqs": 32},
    }


def test_sampled_artifact_requires_exact_sample_index_panel() -> None:
    source_hash = "1" * 64
    rgb_hash = "2" * 64
    prompt_metadata = {
        "example-1": {
            "prompt_token_ids": [1],
            "source_image_file_sha256": source_hash,
        }
    }
    rows = [
        {
            "image_id": 1,
            "example_id": "example-1",
            "trajectory_id": f"sample-{index:02d}",
            "decode_mode": "sampled",
            "sample_index": index,
            "source_image_file_sha256": source_hash,
            "executed_rgb_sha256": rgb_hash,
        }
        for index in range(16)
    ]
    artifact = build_artifact(
        config={"decode_mode": "sampled"},
        model_identity={"model": "source-step-4887"},
        prompt_metadata=prompt_metadata,
        rollouts=rows,
    )
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["rollout_count"] == 16
    rows[-1]["sample_index"] = 14
    with pytest.raises(ValueError, match="sample_index 0..15"):
        build_artifact(
            config={"decode_mode": "sampled"},
            model_identity={"model": "source-step-4887"},
            prompt_metadata=prompt_metadata,
            rollouts=rows,
        )


def test_panel_requires_structurally_validated_execution_model() -> None:
    identity = {
        "mode": "materialized",
        "model_path": "/tmp/model",
    }
    assert resolve_panel_execution_model(
        object(), resolver=lambda _resolved: identity, validator=lambda value: value
    ) == identity
    with pytest.raises(RuntimeError, match="resolved execution-model"):
        resolve_panel_execution_model(
            object(), resolver=lambda _resolved: None, validator=lambda value: value
        )


def test_resume_skips_only_hash_valid_complete_zero_truncation_pair(tmp_path: Path) -> None:
    class Example:
        example_id = "example-1"
        metadata = {"source": {"image_id": 1}}

    model_identity = {"model": "source-step-4887"}
    parts: dict[str, dict[str, str]] = {}
    generation_health: dict[str, dict[str, object]] = {}
    for mode, count in (("greedy", 1), ("sampled", 16)):
        rows = [
            {
                "image_id": 1,
                "example_id": "example-1",
                "decode_mode": mode,
                "sample_index": index if mode == "sampled" else None,
                "generated_token_ids": [7],
                "stop_reason": "im_end",
                "predictions": {"parse_status": "accepted", "predictions": []},
            }
            for index in range(count)
        ]
        value = {
            "schema_version": SCHEMA_VERSION,
            "config": {
                "decode_mode": mode,
                "resolved_fingerprint": "resolved",
                "repetition_penalty": 1.0,
            },
            "model_identity": model_identity,
            "rollouts": rows,
        }
        path = tmp_path / f"{mode}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        parts[mode] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        generation_health[mode] = {
            "completion_count": count,
            "generated_token_count": count,
            "stop_reason_counts": {"im_end": count, "length": 0},
            "natural_closure_count": count,
            "parser_status_counts": {"accepted": count},
            "elapsed_seconds": 1.0,
        }
    entry = {
        "batch_index": 0,
        "image_ids": [1],
        "artifacts": parts,
        "generation_health": generation_health,
    }
    assert _validated_resume_batch(
        worker_root=tmp_path,
        entry=entry,
        expected_batch_index=0,
        expected_examples=[Example()],
        resolved_fingerprint="resolved",
        model_identity=model_identity,
    ) is not None
    (tmp_path / "sampled.json").write_text("{}", encoding="utf-8")
    assert _validated_resume_batch(
        worker_root=tmp_path,
        entry=entry,
        expected_batch_index=0,
        expected_examples=[Example()],
        resolved_fingerprint="resolved",
        model_identity=model_identity,
    ) is None


def test_sampled_only_resume_requires_exact_sampled_artifact(tmp_path: Path) -> None:
    class Example:
        example_id = "example-1"
        metadata = {"source": {"image_id": 1}}

    model_identity = {"model": "source-step-4887"}
    rows = [
        {
            "image_id": 1,
            "example_id": "example-1",
            "decode_mode": "sampled",
            "sample_index": index,
            "generated_token_ids": [7],
            "stop_reason": "im_end",
            "predictions": {"parse_status": "accepted", "predictions": []},
        }
        for index in range(16)
    ]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "decode_mode": "sampled",
            "panel_mode": "sampled_only",
            "resolved_fingerprint": "resolved",
            "repetition_penalty": 1.0,
        },
        "model_identity": model_identity,
        "rollouts": rows,
    }
    path = tmp_path / "sampled.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    entry = {
        "batch_index": 0,
        "image_ids": [1],
        "artifacts": {
            "sampled": {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        },
        "generation_health": {
            "sampled": {
                "completion_count": 16,
                "generated_token_count": 16,
                "stop_reason_counts": {"im_end": 16, "length": 0},
                "natural_closure_count": 16,
                "parser_status_counts": {"accepted": 16},
                "elapsed_seconds": 1.0,
            }
        },
    }
    assert _validated_resume_batch(
        worker_root=tmp_path,
        entry=entry,
        expected_batch_index=0,
        expected_examples=[Example()],
        resolved_fingerprint="resolved",
        model_identity=model_identity,
        decode_modes=("sampled",),
    ) is not None
    entry["artifacts"]["greedy"] = {"path": "greedy.json", "sha256": "unused"}
    assert _validated_resume_batch(
        worker_root=tmp_path,
        entry=entry,
        expected_batch_index=0,
        expected_examples=[Example()],
        resolved_fingerprint="resolved",
        model_identity=model_identity,
        decode_modes=("sampled",),
    ) is None


def test_source_b16_artifact_and_resume_retain_raw_completion_and_projected_receipt(
    tmp_path: Path,
) -> None:
    class Example:
        example_id = "example-1"
        metadata = {"source": {"image_id": 1}}

    source_hash = "1" * 64
    rgb_hash = "2" * 64
    tokenizer = SourceTokenizer({0: _valid_source_row("row-0")})
    token_ids = [0, tokenizer.box_end_token_id, tokenizer.im_end_token_id]
    text, receipt = _source_b16_receipt_for(token_ids, tokenizer, stop_reason="im_end")
    raw_parser = parse_compact_object_box_closed(
        text,
        row_id="example-1:source-b16",
        row_index=0,
        image_width=100,
        image_height=100,
    )
    row = {
        "image_id": 1,
        "example_id": "example-1",
        "trajectory_id": "source-b16",
        "decode_mode": "source_b16",
        "generated_token_ids": token_ids,
        "generated_token_ids_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "generated_text": text,
        "stop_reason": "im_end",
        "image_width": 100,
        "image_height": 100,
        "prompt_token_ids": [1],
        "prompt_token_ids_sha256": "prompt-hash",
        "source_image_file_sha256": source_hash,
        "executed_rgb_sha256": rgb_hash,
        "predictions": raw_parser.to_artifact_dict(),
        "source_b16": receipt,
    }
    model_identity = {"model": "source-step-4887"}
    artifact = build_artifact(
        config={
            "decode_mode": "source_b16",
            "panel_mode": "source_b16",
            "resolved_fingerprint": "resolved",
            "source_b16_row_budget": SOURCE_B16_ROW_BUDGET,
            "repetition_penalty": 1.0,
        },
        model_identity=model_identity,
        prompt_metadata={
            "example-1": {
                "prompt_token_ids": [1],
                "source_image_file_sha256": source_hash,
            }
        },
        rollouts=[row],
    )
    path = tmp_path / "source-b16.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    entry = {
        "batch_index": 0,
        "image_ids": [1],
        "artifacts": {
            "source_b16": {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        },
        "generation_health": {
            "source_b16": _generation_health([row], elapsed_seconds=1.0),
        },
        "source_b16": _source_b16_summary([row]),
    }
    assert _validated_resume_batch(
        worker_root=tmp_path,
        entry=entry,
        expected_batch_index=0,
        expected_examples=[Example()],
        resolved_fingerprint="resolved",
        model_identity=model_identity,
        decode_modes=("source_b16",),
        tokenizer=tokenizer,
        stop_token_id=tokenizer.im_end_token_id,
    ) is not None
    assert _validated_resume_batch(
        worker_root=tmp_path,
        entry=entry,
        expected_batch_index=0,
        expected_examples=[Example()],
        resolved_fingerprint="resolved",
        model_identity=model_identity,
        decode_modes=("source_b16",),
        tokenizer=tokenizer,
        stop_token_id=tokenizer.im_end_token_id,
        expected_repetition_penalty=1.1,
    ) is None

    row["source_b16"]["projected_text"] = "garbage"
    rejected_artifact = build_artifact(
        config=artifact["config"],
        model_identity=model_identity,
        prompt_metadata=artifact["prompt_metadata"],
        rollouts=[row],
    )
    path.write_text(json.dumps(rejected_artifact), encoding="utf-8")
    entry["artifacts"]["source_b16"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _validated_resume_batch(
        worker_root=tmp_path,
        entry=entry,
        expected_batch_index=0,
        expected_examples=[Example()],
        resolved_fingerprint="resolved",
        model_identity=model_identity,
        decode_modes=("source_b16",),
        tokenizer=tokenizer,
        stop_token_id=tokenizer.im_end_token_id,
    ) is None

    row["source_b16"]["projected_text"] = text
    row["source_b16"]["projected_parser_evidence"] = {}
    rejected_artifact = build_artifact(
        config=artifact["config"],
        model_identity=model_identity,
        prompt_metadata=artifact["prompt_metadata"],
        rollouts=[row],
    )
    path.write_text(json.dumps(rejected_artifact), encoding="utf-8")
    entry["artifacts"]["source_b16"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _validated_resume_batch(
        worker_root=tmp_path,
        entry=entry,
        expected_batch_index=0,
        expected_examples=[Example()],
        resolved_fingerprint="resolved",
        model_identity=model_identity,
        decode_modes=("source_b16",),
        tokenizer=tokenizer,
        stop_token_id=tokenizer.im_end_token_id,
    ) is None


def test_source_b16_resume_preserves_ineligible_receipts_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    class Example:
        example_id = "example-1"
        metadata = {"source": {"image_id": 1}}

    tokenizer = SourceTokenizer(
        {index: _valid_source_row(f"row-{index}") for index in range(15)}
    )
    token_ids = [
        token
        for index in range(15)
        for token in (index, tokenizer.box_end_token_id)
    ]
    row = _source_b16_rollout(
        image_id=1,
        example_id="example-1",
        token_ids=token_ids,
        tokenizer=tokenizer,
        stop_reason="length",
    )
    assert row["source_b16"]["status"] == "failed_token_limit_before_budget"
    model_identity = {"model": "source-step-4887"}
    artifact = build_artifact(
        config={
            "decode_mode": "source_b16",
            "panel_mode": "source_b16",
            "resolved_fingerprint": "resolved",
            "source_b16_row_budget": SOURCE_B16_ROW_BUDGET,
            "repetition_penalty": 1.0,
        },
        model_identity=model_identity,
        prompt_metadata={
            "example-1": {
                "prompt_token_ids": [1],
                "source_image_file_sha256": row["source_image_file_sha256"],
            }
        },
        rollouts=[row],
    )
    path = tmp_path / "source-b16-ineligible.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    entry = {
        "batch_index": 0,
        "image_ids": [1],
        "artifacts": {
            "source_b16": {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        },
        "generation_health": {
            "source_b16": _generation_health([row], elapsed_seconds=1.0),
        },
        "source_b16": _source_b16_summary([row]),
    }
    assert entry["source_b16"] == {
        "status_counts": {"failed_token_limit_before_budget": 1},
        "accepted_count": 0,
        "ineligible_count": 1,
        "ineligible_image_ids": [1],
    }
    assert _validated_resume_batch(
        worker_root=tmp_path,
        entry=entry,
        expected_batch_index=0,
        expected_examples=[Example()],
        resolved_fingerprint="resolved",
        model_identity=model_identity,
        decode_modes=("source_b16",),
        tokenizer=tokenizer,
        stop_token_id=tokenizer.im_end_token_id,
    ) is not None

    row["source_b16"]["projected_token_ids_sha256"] = "0" * 64
    corrupted_artifact = build_artifact(
        config=artifact["config"],
        model_identity=model_identity,
        prompt_metadata=artifact["prompt_metadata"],
        rollouts=[row],
    )
    path.write_text(json.dumps(corrupted_artifact), encoding="utf-8")
    entry["artifacts"]["source_b16"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _validated_resume_batch(
        worker_root=tmp_path,
        entry=entry,
        expected_batch_index=0,
        expected_examples=[Example()],
        resolved_fingerprint="resolved",
        model_identity=model_identity,
        decode_modes=("source_b16",),
        tokenizer=tokenizer,
        stop_token_id=tokenizer.im_end_token_id,
    ) is None


def test_source_b16_ineligibility_completes_later_batches_and_keeps_real_errors_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Example:
        def __init__(self, example_id: str, image_id: int) -> None:
            self.example_id = example_id
            self.image_id = image_id
            self.metadata = {"source": {"image_id": image_id}}

    class Receipt:
        def to_artifact_dict(self) -> dict[str, object]:
            return {"backend": "vllm", "effective_settings": {}}

    class Session:
        def __init__(self, tokenizer: SourceTokenizer) -> None:
            self._tokenizer = tokenizer
            self.receipt = Receipt()

        def _im_end_token_id(self) -> int:
            return self._tokenizer.im_end_token_id

    tokenizer = SourceTokenizer(
        {
            **{index: _valid_source_row(f"short-{index}") for index in range(15)},
            100: _valid_source_row("accepted"),
        }
    )
    short_token_ids = [
        token
        for index in range(15)
        for token in (index, tokenizer.box_end_token_id)
    ]
    ineligible = _source_b16_rollout(
        image_id=1,
        example_id="example-1",
        token_ids=short_token_ids,
        tokenizer=tokenizer,
        stop_reason="length",
    )
    accepted = _source_b16_rollout(
        image_id=2,
        example_id="example-2",
        token_ids=[100, tokenizer.box_end_token_id, tokenizer.im_end_token_id],
        tokenizer=tokenizer,
        stop_reason="im_end",
    )
    assert ineligible["source_b16"]["status"] == "failed_token_limit_before_budget"
    assert accepted["source_b16"]["status"] == "accepted_natural_end"

    examples = [Example("example-1", 1), Example("example-2", 2)]
    generation = SimpleNamespace(
        batch_size=32,
        max_new_tokens=2048,
        repetition_penalty=1.0,
        model_dump=lambda *, mode: {"mode": mode},
    )
    resolved = SimpleNamespace(
        config=SimpleNamespace(
            backend=SimpleNamespace(type="vllm"),
            generation=generation,
            data=SimpleNamespace(input_jsonl=str(tmp_path / "input.jsonl")),
            model=SimpleNamespace(dtype="bf16"),
        ),
        fingerprint="resolved",
    )
    session = Session(tokenizer)
    rollout_batches: list[list[dict[str, object]]] = [[ineligible], [accepted]]
    generation_call_count = 0

    @contextmanager
    def fake_open_backend_session(*_: object, **__: object):
        yield session

    def fake_build_requests(
        _config: object, _frontend: object, requested_examples: list[Example]
    ) -> tuple[list[Example], dict[str, dict[str, str]]]:
        return requested_examples, {
            example.example_id: {
                "image_sha256": f"{example.image_id:064x}",
            }
            for example in requested_examples
        }

    def fake_generation(**_: object) -> tuple[tuple[object, ...], tuple[str, ...], float]:
        nonlocal generation_call_count
        generation_call_count += 1
        return (), (), 1.0

    def fake_rollout_rows(**_: object) -> list[dict[str, object]]:
        return rollout_batches.pop(0)

    monkeypatch.setattr(
        "src.config.inference.load_infer_config", lambda _path: resolved
    )
    monkeypatch.setattr("src.data.load_raw_examples", lambda _path: examples)
    monkeypatch.setattr("src.inference.backend.open_backend_session", fake_open_backend_session)
    monkeypatch.setattr(
        "src.inference.runtime.assemble_frontend", lambda *_args, **_kwargs: SimpleNamespace(launch={})
    )
    monkeypatch.setattr(collector, "resolve_panel_execution_model", lambda _resolved: {"id": "m"})
    monkeypatch.setattr(collector, "select_examples", lambda values, _ids: values)
    monkeypatch.setattr(collector, "physical_image_id", lambda example: example.image_id)
    monkeypatch.setattr(collector, "_build_requests", fake_build_requests)
    monkeypatch.setattr(collector, "_run_native_generation", fake_generation)
    monkeypatch.setattr(collector, "_rollout_rows", fake_rollout_rows)
    monkeypatch.setattr(collector, "_record_research_live_decode", lambda **_: None)

    infer_config = tmp_path / "source-b16.yaml"
    infer_config.write_text("test: true\n", encoding="utf-8")
    manifest_path = collector.collect_panel(
        infer_config=infer_config,
        output_root=tmp_path / "completed-output",
        worker_index=0,
        worker_count=1,
        image_batch_size=1,
        image_ids=None,
        max_images=None,
        request_seed=None,
        resume=False,
        source_b16=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert generation_call_count == 2
    assert manifest["status"] == "completed_with_source_b16_ineligible"
    assert [batch["source_b16"] for batch in manifest["batches"]] == [
        {
            "status_counts": {"failed_token_limit_before_budget": 1},
            "accepted_count": 0,
            "ineligible_count": 1,
            "ineligible_image_ids": [1],
        },
        {
            "status_counts": {"accepted_natural_end": 1},
            "accepted_count": 1,
            "ineligible_count": 0,
            "ineligible_image_ids": [],
        },
    ]
    assert manifest["source_b16"] == {
        "status_counts": {
            "accepted_natural_end": 1,
            "failed_token_limit_before_budget": 1,
        },
        "accepted_count": 1,
        "ineligible_count": 1,
        "ineligible_image_ids": [1],
    }

    # Keep the artifact hashes current so resume must reject the tampered
    # semantic receipt, rather than merely noticing a stale file hash.
    for batch in manifest["batches"]:
        artifact_part = batch["artifacts"]["source_b16"]
        artifact_path = manifest_path.parent / artifact_part["path"]
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["rollouts"][0]["source_b16"]["projected_token_ids_sha256"] = "0" * 64
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        artifact_part["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rollout_batches[:] = [[ineligible], [accepted]]
    resume_call_count = generation_call_count
    resumed_manifest_path = collector.collect_panel(
        infer_config=infer_config,
        output_root=tmp_path / "completed-output",
        worker_index=0,
        worker_count=1,
        image_batch_size=1,
        image_ids=None,
        max_images=None,
        request_seed=None,
        resume=True,
        source_b16=True,
    )
    resumed_manifest = json.loads(resumed_manifest_path.read_text(encoding="utf-8"))
    assert generation_call_count == resume_call_count + 2
    assert resumed_manifest["status"] == "completed_with_source_b16_ineligible"
    assert resumed_manifest["source_b16"] == manifest["source_b16"]

    monkeypatch.setattr(
        collector,
        "_run_native_generation",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("engine failure")),
    )
    with pytest.raises(RuntimeError, match="engine failure"):
        collector.collect_panel(
            infer_config=infer_config,
            output_root=tmp_path / "infrastructure-failure",
            worker_index=0,
            worker_count=1,
            image_batch_size=1,
            image_ids=None,
            max_images=1,
            request_seed=None,
            resume=False,
            source_b16=True,
        )

    rollout_batches[:] = [[ineligible]]
    monkeypatch.setattr(collector, "_run_native_generation", fake_generation)
    monkeypatch.setattr(
        collector,
        "_atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("artifact failure")),
    )
    with pytest.raises(RuntimeError, match="artifact failure"):
        collector.collect_panel(
            infer_config=infer_config,
            output_root=tmp_path / "artifact-failure",
            worker_index=0,
            worker_count=1,
            image_batch_size=1,
            image_ids=None,
            max_images=1,
            request_seed=None,
            resume=False,
            source_b16=True,
        )
