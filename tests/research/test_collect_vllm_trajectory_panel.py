from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.collect_vllm_trajectory_panel import (
    SCHEMA_VERSION,
    _materialize_completion,
    _ordered_completions,
    _generation_health,
    _stable_session_identity,
    _validated_resume_batch,
    build_artifact,
    parse_image_ids,
    resolve_panel_execution_model,
    sampling_params_kwargs,
    shard_examples,
)


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


@dataclass
class Completion:
    index: int
    token_ids: list[int]
    finish_reason: str


class Tokenizer:
    def decode(self, values: list[int], **_: object) -> str:
        return ",".join(str(value) for value in values)


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
            "config": {"decode_mode": mode, "resolved_fingerprint": "resolved"},
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
