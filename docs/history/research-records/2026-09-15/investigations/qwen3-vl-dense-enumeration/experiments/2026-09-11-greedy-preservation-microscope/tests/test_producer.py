from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("greedy_preservation_producer", HERE / "producer.py")
assert SPEC is not None and SPEC.loader is not None
producer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(producer)


def test_small_kl_can_flip_a_near_tied_argmax() -> None:
    source_ids = [0]
    reference_logits = torch.tensor([[0.0008, 0.0, -12.0]], dtype=torch.float32)
    current_logits = torch.tensor([[0.0, 0.0008, -12.0]], dtype=torch.float32)
    reference = producer.reference_snapshot(reference_logits, source_ids, [0])
    compared = producer.compare_snapshot(
        current_logits, source_ids, [0], reference["protected_logp"], reference["positions"]
    )

    row = compared["positions"][0]
    assert row["reference_source_is_argmax"] is True
    assert row["current_source_is_argmax"] is False
    assert row["source_argmax_retained"] is False
    assert row["reference_source_margin"] == pytest.approx(0.0008)
    assert row["current_source_margin"] == pytest.approx(-0.0008)
    assert row["near_tie_reference"] is True and row["near_tie_current"] is True
    assert 0.0 <= row["reference_kl"] < 1e-5


def test_literal_source_exception_is_not_relabelled_as_a_checkpoint_flip() -> None:
    source_ids = [0]
    reference_logits = torch.tensor([[0.0, 0.2]], dtype=torch.float32)
    current_logits = torch.tensor([[0.0, 0.3]], dtype=torch.float32)
    reference = producer.reference_snapshot(reference_logits, source_ids, [0])
    compared = producer.compare_snapshot(
        current_logits, source_ids, [0], reference["protected_logp"], reference["positions"]
    )
    row = compared["positions"][0]
    assert row["reference_source_is_argmax"] is False
    assert row["current_source_is_argmax"] is False
    assert row["source_argmax_retained"] is None


def test_first_divergence_preserves_identical_and_prefix_cases() -> None:
    assert producer.first_divergence([1, 2, 3], [1, 4, 3]) == {
        "position": 1,
        "common_prefix_length": 1,
        "shared_prefix_ids_sha256": producer.digest_ids([1]),
        "old_token_id": 2,
        "new_token_id": 4,
        "kind": "token_substitution",
    }
    assert producer.first_divergence([1, 2], [1, 2]) is None
    assert producer.first_divergence([1, 2], [1, 2, 3]) == {
        "position": 2,
        "common_prefix_length": 2,
        "shared_prefix_ids_sha256": producer.digest_ids([1, 2]),
        "old_token_id": None,
        "new_token_id": 3,
        "kind": "old_ended",
    }


def test_real_input_join_has_frozen_population_masks_and_endpoint() -> None:
    packet = producer.assemble_input_packet()
    producer.validate_input_packet(packet, verify_sources=True)

    assert packet["counts"] == {
        "cases": 56,
        "action_tokens": 6056,
        "protected_positions": 6047,
        "workers": 8,
        "cases_per_worker": 7,
    }
    assert [len(shard) for shard in packet["shards"]] == [7] * 8
    assert len({key for shard in packet["shards"] for key in shard}) == 56
    assert max(row["estimated_reference_cache_bytes"] for row in packet["shard_plan"]) == 841_517_040
    assert sum(row["action_tokens"] for row in packet["shard_plan"]) == 6056
    assert packet["geometry_invalid_exclusion"] == {
        "image_id": "360573",
        "positions": [75, 76, 77, 78, 79, 80, 81, 82, 83],
    }
    assert packet["endpoint_summary"]["iou50"] == {
        "stable_tp": 416,
        "positive32_tp": 389,
        "lost_owners": 33,
        "gained_owners": 6,
    }
    assert packet["endpoint_summary"]["burden"] == {
        "stable_strict_repeats": 0,
        "positive32_strict_repeats": 7,
        "stable_parser_drops": 1,
        "positive32_parser_drops": 2,
    }
    assert packet["endpoint_summary"]["identical_natural_actions"] == 1


def test_input_validator_rejects_changed_case_alignment() -> None:
    packet = producer.assemble_input_packet()
    packet["cases"][0]["natural"]["current_action_ids"][0] += 1
    packet["cases"][0]["natural"]["current_action_ids_sha256"] = producer.digest_ids(
        packet["cases"][0]["natural"]["current_action_ids"]
    )
    content = dict(packet)
    content.pop("content_sha256")
    packet["content_sha256"] = producer.digest_json(content)
    with pytest.raises(ValueError, match="natural first divergence"):
        producer.validate_input_packet(packet, verify_sources=False)


def test_resource_reduction_keeps_lifecycle_peaks() -> None:
    observed = producer.combine_resource_observations([
        {
            "phase": "after_stable50",
            "peak_cuda_allocated_bytes": 12,
            "peak_cuda_reserved_bytes": 14,
            "peak_rss_bytes": 30,
            "elapsed_seconds": 3.0,
        },
        {
            "phase": "terminal",
            "peak_cuda_allocated_bytes": 9,
            "peak_cuda_reserved_bytes": 11,
            "peak_rss_bytes": 35,
            "elapsed_seconds": 5.0,
        },
    ])
    assert observed["peak_cuda_allocated_bytes"] == 12
    assert observed["peak_cuda_reserved_bytes"] == 14
    assert observed["peak_rss_bytes"] == 35
    assert observed["elapsed_seconds"] == 5.0


def test_consumer_preserves_shared_prefix_exceptions_and_exact_denominators() -> None:
    packet = producer.assemble_input_packet()
    records = []
    for case in packet["cases"]:
        source = case["source_case"]
        protected = set(source["initial_layout"]["kl_positions"])
        positions = []
        for position, token_id in enumerate(source["action_ids"]):
            positions.append({
                "position": position,
                "source_token_id": token_id,
                "protected": position in protected,
                "reference_kl": 0.0 if position in protected else None,
                "reference_source_logprob": -0.1,
                "current_source_logprob": -0.1,
                "reference_source_margin": 0.1,
                "current_source_margin": 0.1,
                "reference_top1_id": token_id,
                "current_top1_id": token_id,
                "reference_source_is_argmax": True,
                "current_source_is_argmax": True,
                "source_argmax_retained": True,
                "near_tie_reference": False,
                "near_tie_current": False,
                "source_token": "token",
                "token_category": "description_or_content",
            })
        divergence = case["natural"]["first_divergence"]
        if divergence is not None and divergence["position"] < len(positions):
            scored = positions[divergence["position"]]
            # Keep one literal-source exception and one checkpoint flip visible.
            if not records:
                scored["reference_source_is_argmax"] = False
                scored["source_argmax_retained"] = None
            elif len(records) == 1:
                scored["current_source_is_argmax"] = False
                scored["source_argmax_retained"] = False
                scored["current_top1_id"] = divergence["new_token_id"]
        records.append({
            "key": case["key"], "image_id": case["image_id"],
            "positions": positions, "natural": case["natural"],
        })
    summary = producer.summarize_records(records, packet)
    assert summary["counts"] == {
        "cases": 56, "action_tokens": 6056, "protected_positions": 6047,
        "natural_divergences": 55, "identical_natural_actions": 1,
    }
    assert summary["greedy_margin"]["reference_literal_argmax_exceptions"] == 1
    assert summary["greedy_margin"]["checkpoint_argmax_flips"] == 1
    assert len(summary["first_natural_divergences"]) == 55


def test_token_category_is_explicit_without_retokenization() -> None:
    assert producer.token_category("<|coord_14|>") == "coordinate"
    assert producer.token_category("<|object_ref_start|>") == "object_ref_start"
    assert producer.token_category("<|box_end|>") == "box_end"
    assert producer.token_category("person") == "description_or_content"
