from __future__ import annotations

import copy

import pytest

from scripts.research import run_continuation_locality_boundary_scoring as locality
from scripts.research import run_exact_prefix_owner_compositionality as owner_probe
from scripts.research import summarize_continuation_locality_owner_compositionality as summarize
from scripts.research.materialize_continuation_locality_owner_compositionality import (
    SCHEMA_VERSION,
    _row_prefixes,
)
from src.inference.backend import token_ids_sha256


def _complete_row(description: list[int]) -> list[int]:
    return [151646, *description, 151647, 151648, 151670, 151671, 151672, 151673, 151649]


def test_description_force_keeps_the_full_multi_token_description() -> None:
    row = _complete_row([41, 42, 43])

    assert owner_probe._description_force(row) == [151646, 41, 42, 43, 151647]


def test_complete_row_prefixes_preserve_every_literal_boundary() -> None:
    first = _complete_row([51])
    second = _complete_row([52, 53])

    assert _row_prefixes(first + second) == [first, first + second]


def test_locality_and_owner_runners_use_the_same_stable_sharding() -> None:
    values = ["case-a", "case-b", "image-225458-depth-1-owner-bottle"]

    for value in values:
        assert locality._stable_shard(value, 7) == owner_probe._stable_shard(value, 7)


def test_owner_manifest_validates_composite_owner_ids_and_literal_hashes() -> None:
    prompt = [101, 102]
    prefix = _complete_row([61])
    target_row = _complete_row([62, 63])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "owner_cases": [
            {
                "case_id": "case-a",
                "base_prompt_token_ids": prompt,
                "base_prompt_token_ids_sha256": token_ids_sha256(prompt),
                "prefix_token_ids": prefix,
                "prefix_token_ids_sha256": token_ids_sha256(prefix),
                "target": {
                    "owner_id": "225458:1234",
                    "category": "bottle",
                    "row_token_ids": target_row,
                    "row_token_ids_sha256": token_ids_sha256(target_row),
                },
                "secondary_targets": [],
            }
        ],
    }

    cases = owner_probe._validated_cases(manifest)
    assert cases[0]["target"]["owner_id"] == "225458:1234"

    corrupted = copy.deepcopy(manifest)
    corrupted["owner_cases"][0]["target"]["row_token_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="row hash mismatch"):
        owner_probe._validated_cases(corrupted)


def test_binary_recovery_transitions_preserve_pairing() -> None:
    assert summarize._binary_transition_counts(
        [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]
    ) == {
        "false_to_false": 1,
        "false_to_true": 1,
        "true_to_false": 1,
        "true_to_true": 1,
    }
