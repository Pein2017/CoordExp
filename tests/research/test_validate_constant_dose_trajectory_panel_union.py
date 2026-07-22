from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.validate_constant_dose_trajectory_panel_union import (
    DEFAULT_PRODUCER_SCRIPT,
    FROZEN_PRODUCER_SCRIPT_SHA256,
    validate_panel_union,
)
from scripts.research.split_label_only_candidate_pool import (
    split_label_only_candidate_pool,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_ids_sha256(rows: list[dict[str, object]]) -> str:
    encoded = "".join(f"{row['image_id']}\n" for row in rows).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_panel(path: Path, ids: range, *, mode: str, duplicate: bool = False) -> None:
    seeds = [31000] if mode == "greedy" else list(range(31001, 31017))
    rows = [
        {
            "image_id": image_id,
            "example_id": f"example-{image_id}",
            "seed": seed,
            "decode_mode": mode,
        }
        for image_id in ids
        for seed in seeds
    ]
    if duplicate:
        rows.append(dict(rows[0]))
    prompt_metadata: dict[str, dict[str, object]] = {}
    for image_id in ids:
        # The varying image-pad lengths intentionally make prompt-token hashes
        # unsuitable as the shared prompt-policy fingerprint.
        prompt_ids = [1, *([151655] * (image_id % 4 + 1)), 2]
        prompt_metadata[f"example-{image_id}"] = {
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": _sha256_json(prompt_ids),
            "chat_text_sha256": "a" * 64,
        }
    path.write_text(
        json.dumps(
            {
                "schema_version": "current_seeded_sampled_rollouts.v1",
                "config": {
                    "decode_mode": mode,
                    "sampling_is_not_infer_config": True,
                    "temperature": 0.0 if mode == "greedy" else 0.4,
                    "top_p": 1.0 if mode == "greedy" else 0.95,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": 512,
                    "seeds": seeds,
                },
                "model_identity": {"checkpoint": "source-4887"},
                "prompt_metadata": prompt_metadata,
                "rollouts": rows,
            }
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path, *, duplicate: bool = False) -> dict[str, Path]:
    pool = tmp_path / "candidate-pool.jsonl"
    object_counts = (1, 4, 8, 16)
    pool.write_text(
        "".join(
            json.dumps(
                {
                    "image_id": image_id,
                    "objects": [
                        {"index": index} for index in range(object_counts[image_id % 4])
                    ],
                }
            )
            + "\n"
            for image_id in range(2432)
        ),
        encoding="utf-8",
    )
    split_dir = tmp_path / "split"
    split_label_only_candidate_pool(input_path=pool, output_dir=split_dir)
    paths = {
        "candidate_pool": pool,
        "split_receipt": split_dir / "split-receipt.json",
    }
    for panel, ids, mode in (
        ("old_greedy", range(256), "greedy"),
        ("old_sampled", range(256), "sampled"),
        ("new_greedy", range(256, 2432), "greedy"),
        ("new_sampled", range(256, 2432), "sampled"),
    ):
        path = tmp_path / f"{panel}.json"
        _write_panel(
            path, ids, mode=mode, duplicate=duplicate and panel == "new_sampled"
        )
        paths[panel] = path
    return paths


def _validate(
    paths: dict[str, Path], *, producer_script: Path | None = None
) -> dict[str, object]:
    return validate_panel_union(
        candidate_pool=paths["candidate_pool"],
        split_receipt=paths["split_receipt"],
        old_greedy=[paths["old_greedy"]],
        old_sampled=[paths["old_sampled"]],
        new_greedy=[paths["new_greedy"]],
        new_sampled=[paths["new_sampled"]],
        producer_script=producer_script,
    )


def _artifact(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(path: Path, artifact: dict[str, object]) -> None:
    path.write_text(json.dumps(artifact), encoding="utf-8")


def _swap_split_rows(paths: dict[str, Path], *, same_band: bool) -> None:
    receipt = _artifact(paths["split_receipt"])
    outputs = receipt["outputs"]
    assert isinstance(outputs, dict)
    train_entry = outputs["train_candidate"]
    development_entry = outputs["development"]
    assert isinstance(train_entry, dict) and isinstance(development_entry, dict)
    train_path = Path(str(train_entry["path"]))
    development_path = Path(str(development_entry["path"]))
    train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    development_rows = [
        json.loads(line) for line in development_path.read_text().splitlines()
    ]
    pair = next(
        (train_index, development_index)
        for train_index, train_row in enumerate(train_rows)
        for development_index, development_row in enumerate(development_rows)
        if (int(train_row["image_id"]) % 4 == int(development_row["image_id"]) % 4)
        is same_band
    )
    train_index, development_index = pair
    train_rows[train_index], development_rows[development_index] = (
        development_rows[development_index],
        train_rows[train_index],
    )
    for split_path, rows, entry in (
        (train_path, train_rows, train_entry),
        (development_path, development_rows, development_entry),
    ):
        split_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        entry["sha256"] = _sha256(split_path)
        entry["ordered_image_ids_sha256"] = _ordered_ids_sha256(rows)
    _write_artifact(paths["split_receipt"], receipt)


def test_validates_exact_2432_image_union_and_split(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    result = _validate(paths)
    assert result["passed"] is True
    assert result["image_count"] == 2432
    assert result["prompt_policy_fingerprint"] == "a" * 64
    assert result["execution_metadata"] == {
        "physical_batch_size": 1,
        "sampling_order": "request_major",
        "rng_reset": "per_image_seed",
        "producer_script": str(DEFAULT_PRODUCER_SCRIPT.resolve()),
        "producer_script_sha256": FROZEN_PRODUCER_SCRIPT_SHA256,
    }


def test_rejects_duplicate_image_seed_pair(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate image-seed pair"):
        _validate(paths)


def test_rejects_artifact_without_request_scoped_sampling_attestation(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    artifact = _artifact(paths["new_sampled"])
    artifact["config"].pop("sampling_is_not_infer_config")
    _write_artifact(paths["new_sampled"], artifact)
    with pytest.raises(ValueError, match="request-scoped sampling"):
        _validate(paths)


def test_rejects_wrong_decode_config_fields_and_seeds(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    mutations = (
        ("new_sampled", "temperature", 0.5, "temperature"),
        ("new_sampled", "top_p", 0.9, "top_p"),
        ("new_sampled", "repetition_penalty", 1.1, "repetition_penalty"),
        ("new_sampled", "max_new_tokens", 511, "max_new_tokens"),
        ("new_sampled", "seeds", [*range(31001, 31016), 31999], "sampled seeds"),
        ("new_greedy", "temperature", 0.1, "temperature"),
        ("new_greedy", "top_p", 0.9, "top_p"),
        ("new_greedy", "repetition_penalty", 1.1, "repetition_penalty"),
        ("new_greedy", "max_new_tokens", 511, "max_new_tokens"),
        ("new_greedy", "seeds", [31001], "seed 31000"),
    )
    for panel, field, replacement, error in mutations:
        original = _artifact(paths[panel])
        mutated = _artifact(paths[panel])
        config = mutated["config"]
        assert isinstance(config, dict)
        config[field] = replacement
        _write_artifact(paths[panel], mutated)
        with pytest.raises(ValueError, match=error):
            _validate(paths)
        _write_artifact(paths[panel], original)


def test_rejects_wrong_producer_sha(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    producer = tmp_path / "producer.py"
    producer.write_text("# unaudited producer\n", encoding="utf-8")
    with pytest.raises(ValueError, match="producer script SHA"):
        _validate(paths, producer_script=producer)


def test_rejects_chat_text_prompt_policy_drift(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    artifact = _artifact(paths["new_sampled"])
    metadata = artifact["prompt_metadata"]
    assert isinstance(metadata, dict)
    entry = next(iter(metadata.values()))
    assert isinstance(entry, dict)
    entry["chat_text_sha256"] = "b" * 64
    _write_artifact(paths["new_sampled"], artifact)
    with pytest.raises(ValueError, match="chat-text prompt policy"):
        _validate(paths)


@pytest.mark.parametrize("field", ["prompt_token_ids", "prompt_token_ids_sha256"])
def test_rejects_incomplete_exact_prompt_id_evidence(
    tmp_path: Path, field: str
) -> None:
    paths = _inputs(tmp_path)
    artifact = _artifact(paths["new_sampled"])
    metadata = artifact["prompt_metadata"]
    assert isinstance(metadata, dict)
    entry = next(iter(metadata.values()))
    assert isinstance(entry, dict)
    entry.pop(field)
    _write_artifact(paths["new_sampled"], artifact)
    with pytest.raises(ValueError, match="exact prompt IDs|prompt-ID hash mismatch"):
        _validate(paths)


def test_rejects_non_request_major_row_order(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    artifact = _artifact(paths["new_sampled"])
    rows = artifact["rollouts"]
    assert isinstance(rows, list)
    rows[0], rows[16] = rows[16], rows[0]
    _write_artifact(paths["new_sampled"], artifact)
    with pytest.raises(ValueError, match="contiguous request-major"):
        _validate(paths)


def test_rejects_arbitrary_same_band_split_membership(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    _swap_split_rows(paths, same_band=True)
    with pytest.raises(ValueError, match="deterministic seed-19 membership"):
        _validate(paths)


def test_rejects_split_band_drift(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    _swap_split_rows(paths, same_band=False)
    with pytest.raises(ValueError, match="wrong per-band membership"):
        _validate(paths)
