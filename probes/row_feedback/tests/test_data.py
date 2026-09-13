import copy

import pytest

from probes.row_feedback import data


def test_exact_admitted_n16_bank_has_literal_h_c_to_w_and_required_floor():
    bank = data.build_bank()
    assert len(bank["records"]) == 16
    assert len(bank["train_image_ids"]) == 11
    assert bank["denominators"]["unknown_or_unmatched_as_negative"] == 0
    for record in bank["records"]:
        assert record["visible_history_token_ids"] == (
            record["literal_rows"]["h"]["token_ids"] + record["literal_rows"]["c"]["token_ids"]
        )
        assert record["visible_target_token_ids"] == record["literal_rows"]["w"]["token_ids"]
        assert record["teacher"]["mapping"] == "visible_target_ordinal"
        assert record["literal_rows"]["h"]["role"] == "vetted_older_anchor_rollout_history"
        assert record["prefix_w_provenance"]["rollout_adapter_fingerprint"] != \
            record["teacher"]["source_adapter"]["fingerprint"]
        assert record["supervision_groups"]["entry_c"]["native_teacher_protection"] == \
            "forbidden_supervised_repaired_target"
        assert record["supervision_groups"]["post_completion_w"]["visible_history_token_ids"] == \
            record["visible_history_token_ids"]


def test_validation_rejects_target_or_hash_mutation():
    bank = data.build_bank()
    mutated = copy.deepcopy(bank)
    mutated["records"][0]["visible_target_token_ids"] = list(
        mutated["records"][0]["visible_target_token_ids"]
    )
    mutated["records"][0]["visible_target_token_ids"][1] += 1
    with pytest.raises(ValueError, match="target must be literal post-c w"):
        data.validate_bank(mutated)

    mutated = copy.deepcopy(bank)
    mutated["records"][0]["record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="record hash"):
        data.validate_bank(mutated)


def test_validation_rejects_source_producer_teacher_role_conflation():
    bank = data.build_bank()
    mutated = copy.deepcopy(bank)
    mutated["source_roles"]["prefix_w_rollout_producer"]["adapter"] = copy.deepcopy(
        mutated["source_roles"]["fit_anchor"]["adapter"]
    )
    without_hash = dict(mutated)
    without_hash.pop("bank_sha256")
    mutated["bank_sha256"] = data._json_digest(without_hash)
    with pytest.raises(ValueError, match="older rollout producer must differ from fit anchor"):
        data.validate_bank(mutated)


def test_protection_records_bind_exact_masks_to_fresh_n16_not_stable50():
    packet = data.build_protection_records()
    assert packet["denominators"] == {
        "records": 54, "distinct_images": 54, "visible_action_tokens": 5768,
        "protected_positions": 5759,
    }
    assert packet["train_image_overlap"] == {"count": 0, "example_ids": [],
                                             "policy": "reported_only; selected54 masks and IDs are unchanged"}
    assert packet["fresh_n16_teacher"]["fingerprint"] == "a7dcb56ea71ee8ab37a944778b7947c78ca22dfd321a0dcc59dde5227acecc80"

    mutated = copy.deepcopy(packet)
    mutated["records"][0]["kl_positions"] = mutated["records"][0]["kl_positions"][:-1]
    without_hash = dict(mutated)
    without_hash.pop("protection_records_sha256")
    mutated["protection_records_sha256"] = data._json_digest(without_hash)
    with pytest.raises(ValueError, match="KL mask drift"):
        data.validate_protection_records(mutated)
