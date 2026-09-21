from __future__ import annotations

import copy

import pytest

from probes.dora_owner_learning import margin_preserved_endpoint as accepted
from probes.dora_owner_learning import positive_progress_matched_endpoint as endpoint


def test_real_selection_and_retained_ledgers_are_exact() -> None:
    selection = endpoint.validate_selection()
    old_packet = endpoint.load_old_packet()
    retained = endpoint.validate_retained_ledgers(old_packet)

    assert selection["optimizer_updates"] == 17
    assert selection["selected_scalar_step"] == 18
    assert selection["selected"]["adapter_state_sha256"] == endpoint.SELECTED_STATE_SHA256
    assert selection["selected"]["sum_positive_nll"] == pytest.approx(21.453418254852295)
    assert selection["target_C32_sum_positive_nll"] == pytest.approx(21.61498737335205)
    assert retained == {
        "A_natural": 384,
        "C_natural": 384,
        "C_conditional": 6,
        "C_arm": "C",
        "C_result_sha256": endpoint.C_RESULT_SHA256,
    }


def test_selection_fails_closed_on_k_state_or_natural_field_change() -> None:
    original = endpoint.load_json(endpoint.SELECTION)
    for key, value, message in (
        ("optimizer_updates", 18, "selected k17"),
        ("natural_endpoint_fields_used", True, "natural endpoint"),
    ):
        changed = copy.deepcopy(original)
        changed[key] = value
        with pytest.raises(ValueError, match=message):
            endpoint.validate_selection_payload(changed)
    changed = copy.deepcopy(original)
    changed["selected"]["adapter_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="selected A state"):
        endpoint.validate_selection_payload(changed)


def test_d_consumer_fixture_rejects_c_label_order_geometry_and_score() -> None:
    packet = endpoint.packet_base(endpoint.load_old_packet())
    rows = copy.deepcopy(endpoint.load_json(endpoint.C_CONSUMER))
    for row in rows:
        row["arm"] = "D"
        row["schema"] = endpoint.NATURAL_SCHEMA
    endpoint.validate_d_natural_rows(rows, packet)

    wrong = copy.deepcopy(rows)
    wrong[0]["arm"] = "C"
    with pytest.raises(ValueError, match="only D"):
        endpoint.validate_d_natural_rows(wrong, packet)
    wrong = copy.deepcopy(rows)
    wrong[0], wrong[1] = wrong[1], wrong[0]
    with pytest.raises(ValueError, match="ordered shard"):
        endpoint.validate_d_natural_rows(wrong, packet)
    wrong = copy.deepcopy(rows)
    wrong[0]["executed_media_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt/media/grid"):
        endpoint.validate_d_natural_rows(wrong, packet)
    wrong = copy.deepcopy(rows)
    wrong[0]["score"]["50"]["tp"] += 1
    with pytest.raises(ValueError, match="score"):
        endpoint.validate_d_natural_rows(wrong, packet)


def test_d_conditional_fixture_preserves_forced_free_partition() -> None:
    packet = endpoint.packet_base(endpoint.load_old_packet())
    rows = copy.deepcopy(endpoint.load_json(endpoint.C_CONDITIONAL))
    for row in rows:
        row["arm"] = "D"
        row["schema"] = endpoint.CONDITIONAL_SCHEMA
    endpoint.validate_d_conditional_rows(rows, packet)
    assert [row["credit_identity"]["forced_candidate_row_count"] for row in rows] == [0, 1, 0, 1, 0, 1]
    assert all(not row["credit_identity"]["forced_candidate_in_free_counts"] for row in rows)

    rows[1]["credit_identity"]["forced_candidate_in_free_counts"] = True
    with pytest.raises(ValueError, match="credit identity"):
        endpoint.validate_d_conditional_rows(rows, packet)


def test_endpoint_reuses_accepted_hard_budget_and_package_imports_trainer() -> None:
    from probes.dora_owner_learning import positive_progress_matched_train as trainer

    packet = endpoint.packet_base(endpoint.load_old_packet())
    bounds = packet["resource_bounds"]
    assert endpoint.forward_budget_hook is accepted.forward_budget_hook
    assert endpoint.ForwardBudgetExceeded is accepted.ForwardBudgetExceeded
    assert all(row["max_generated_tokens"] == row["max_model_forwards"] == 15_000 for row in bounds["ranks"])
    assert sum(row["max_generated_tokens"] for row in bounds["ranks"]) == 120_000
    assert trainer.__package__ == "probes.dora_owner_learning"
    assert callable(trainer.verify_receipt)


def test_d_training_metadata_cannot_admit_c() -> None:
    with pytest.raises(ValueError, match="only D"):
        endpoint.validate_d_training_metadata({"arm": "C"}, {}, endpoint.validate_selection())


def test_primary_owner_orientation_is_d_after_c_before() -> None:
    c = {"50": {"owners": ["a", "b"]}, "60": {"owners": ["a"]}, "80": {"owners": []}}
    d = {"50": {"owners": ["b", "c"]}, "60": {"owners": []}, "80": {"owners": ["d"]}}
    assert endpoint.d_vs_c_owner_counts([c], [d]) == {
        "50": {"gained": 1, "lost": 1, "retained": 1},
        "60": {"gained": 0, "lost": 1, "retained": 0},
        "80": {"gained": 1, "lost": 0, "retained": 0},
    }
