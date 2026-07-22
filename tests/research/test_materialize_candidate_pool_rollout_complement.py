from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.materialize_candidate_pool_rollout_complement import (
    materialize_candidate_pool_rollout_complement,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_materializes_byte_preserving_ordered_complement_and_immutable_receipt(
    tmp_path: Path,
) -> None:
    candidate_pool = tmp_path / "candidate-pool.jsonl"
    raw_rows = [
        b'{"image_id":11,"note":"first"}\n',
        b'{ "image_id" : 12, "note" : "second" }\n',
        b'{"image_id":13,"note":"third"}\n',
        b'{"image_id":14,"note":"fourth"}\n',
    ]
    candidate_pool.write_bytes(b"".join(raw_rows))
    rollout = tmp_path / "greedy.json"
    rollout.write_text(
        json.dumps(
            {
                "schema_version": "current_seeded_sampled_rollouts.v1",
                "rollouts": [{"image_id": 11}, {"image_id": "13"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "complement.jsonl"
    receipt = tmp_path / "receipt.json"

    document = materialize_candidate_pool_rollout_complement(
        candidate_pool=candidate_pool,
        existing_rollout_artifact=rollout,
        output_jsonl=output,
        receipt=receipt,
        expected_pool_count=4,
        expected_existing_count=2,
        expected_complement_count=2,
    )

    assert output.read_bytes() == raw_rows[1] + raw_rows[3]
    assert document["counts"] == {
        "candidate_pool": 4,
        "existing_rollout_images": 2,
        "complement": 2,
    }
    assert document["output_jsonl"]["output_row_sha256"] == _sha256(
        raw_rows[1] + raw_rows[3]
    )
    persisted = json.loads(receipt.read_text(encoding="utf-8"))
    assert persisted == document

    with pytest.raises(FileExistsError, match="immutable output already exists"):
        materialize_candidate_pool_rollout_complement(
            candidate_pool=candidate_pool,
            existing_rollout_artifact=rollout,
            output_jsonl=output,
            receipt=receipt,
        )


def test_relative_image_paths_require_output_beside_candidate_pool(
    tmp_path: Path,
) -> None:
    candidate_pool = tmp_path / "candidate-pool.jsonl"
    candidate_pool.write_text(
        json.dumps({"image_id": 11, "images": ["../../images/eleven.jpg"]})
        + "\n",
        encoding="utf-8",
    )
    rollout = tmp_path / "greedy.json"
    rollout.write_text(
        json.dumps(
            {
                "schema_version": "current_seeded_sampled_rollouts.v1",
                "rollouts": [{"image_id": 11}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be written beside"):
        materialize_candidate_pool_rollout_complement(
            candidate_pool=candidate_pool,
            existing_rollout_artifact=rollout,
            output_jsonl=tmp_path / "nested" / "complement.jsonl",
            receipt=tmp_path / "nested" / "receipt.json",
        )
