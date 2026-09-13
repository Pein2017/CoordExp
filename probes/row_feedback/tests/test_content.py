from __future__ import annotations

import json

import pytest
import torch

from probes.row_feedback import content


def _receipt(value: str, *, shape=(4,), dtype="torch.float32") -> dict:
    return {"shape": list(shape), "dtype": dtype, "sha256": value, "finite": True, "rms": 1.0}


def test_packet_binds_three_reviewed_exposed_cases_and_boundaries() -> None:
    packet = content.build_packet()

    assert packet["status"] == "candidate_pending_endpoint_binding"
    assert packet["schema"] == "row_feedback.content_packet.v2"
    assert packet["provenance_boundary"]["literal_history_and_w_producer"]["adapter_fingerprint"] == content.OLDER_HISTORY_PRODUCER
    assert packet["provenance_boundary"]["fit_anchor"]["adapter_fingerprint"] == content.N16_FIT_ANCHOR
    assert [case["image"]["image_id"] for case in packet["cases"]] == [210457, 219546, 417044]
    assert [case["recipient"]["reviewed_c"]["category"] for case in packet["cases"]] == ["cup", "spoon", "donut"]
    for case in packet["cases"]:
        recipient = case["recipient"]
        donor = case["wrong_owner_donor"]
        assert recipient["history_token_ids"][-1] == content.BOX_END
        assert donor["replay_target_token_ids"][-1] == content.BOX_END
        assert recipient["feedback_boundary"]["visible_boundary_index"] == len(recipient["history_token_ids"]) - 1
        assert donor["feedback_boundary"]["visible_boundary_index"] == (
            len(donor["replay_history_token_ids"]) + len(donor["replay_target_token_ids"]) - 1
        )
        assert donor["category"] == recipient["reviewed_c"]["category"]
        assert case["case_provenance"]["history_and_w_rollout_producer_adapter_fingerprint"] == content.OLDER_HISTORY_PRODUCER
        assert case["same_owner_alternate_geometry"]["use"] is False
        plan = content.runtime_call_plan(case)
        assert plan["recipient_boundary"] == recipient["feedback_boundary"]
        assert plan["donor_boundary"] == donor["feedback_boundary"]
        assert plan["correct_f"]["max_visible_tokens"] == plan["replay"]["max_visible_tokens"] == 3084

    content.verify_packet(packet)


def test_packet_rejects_endpoint_overlap(tmp_path) -> None:
    endpoint = tmp_path / "selection.json"
    endpoint.write_text(json.dumps({"image_ids": [210457, 1, 2]}), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps immutable endpoint"):
        content.build_packet(endpoint_selection=endpoint)


def test_packet_freezes_after_disjoint_endpoint_binding(tmp_path) -> None:
    endpoint = tmp_path / "selection.json"
    endpoint.write_text(json.dumps({"image_ids": [1, 2, 3]}), encoding="utf-8")
    packet = content.build_packet(endpoint_selection=endpoint)
    assert packet["status"] == "frozen_ready_for_runtime"
    assert packet["endpoint_binding"]["status"] == "bound_disjoint"
    content.verify_packet(packet)


def test_tensor_receipt_distinguishes_equal_norm_sources() -> None:
    left = content.tensor_receipt(torch.tensor([1.0, -1.0]))
    right = content.tensor_receipt(torch.tensor([-1.0, 1.0]))
    assert left["shape"] == right["shape"]
    assert left["rms"] == right["rms"] == 1.0
    assert left["sha256"] != right["sha256"]


def test_mechanical_classification_separates_validity_dependence_and_semantics() -> None:
    valid = dict(
        correct_visible_ids=[1, 2],
        self_visible_ids=[1, 2],
        wrong_visible_ids=[1, 3],
        self_override_count=1,
        wrong_override_count=1,
        correct_source_receipt=_receipt("c"),
        wrong_source_receipt=_receipt("w"),
    )
    assert content.classify_mechanical_outcome(**valid)["claim"] == "route_dependence_only_pending_or_nonspecific_physical_review"
    assert content.classify_mechanical_outcome(**valid, physical_review="selective_owner_consistent")["claim"] == "supports_content_use_by_the_trained_F_route"

    no_effect = {**valid, "wrong_visible_ids": [1, 2]}
    assert content.classify_mechanical_outcome(**no_effect)["claim"] == "inconclusive_for_memory_and_architecture"

    invalid = {**valid, "self_visible_ids": [9], "self_override_count": 0}
    result = content.classify_mechanical_outcome(**invalid)
    assert result["status"] == "technical_invalid"
    assert set(result["reasons"]) == {"override_count", "self_replay_mismatch"}
