"""Focused contracts for the duplicate-trajectory StateBank draft assembler."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.research.assemble_duplicate_trajectory_state_banks import (
    AssemblyError,
    BANK_NAMES,
    CLEANED,
    COMBINED,
    LOCAL,
    POSITIVE_ONLY,
    SOURCE_ONLY,
    build_duplicate_trajectory_bank_drafts,
    exact_row_site_types,
    exact_row_slices,
    materialize_duplicate_trajectory_state_banks,
    state_bank_rows_from_duplicate_draft_bank,
    write_duplicate_trajectory_bank_drafts,
)
from src.config.fingerprint import sha256_file, sha256_json
from src.inference.backend import token_ids_sha256


def _row(description_token: int, coordinate_start: int) -> list[int]:
    """One deliberately non-textual exact row; no decode path is available."""

    return [
        151646,
        description_token,
        151647,
        151648,
        coordinate_start,
        coordinate_start + 1,
        coordinate_start + 2,
        coordinate_start + 3,
        151649,
    ]


def _review(
    trajectory_id: str,
    row_index: int,
    owner_id: str | None,
    role: str,
    *,
    burst_id: str | None = None,
    category: str | None = None,
) -> dict[str, object]:
    usable = role in {"accepted", "recovery"}
    trusted = role in {"accepted", "recovery", "duplicate"}
    return {
        "trajectory_id": trajectory_id,
        "row_index": row_index,
        "physical_owner_id": owner_id,
        "review_role": role,
        "usable_owner_row": usable,
        "burst_id": burst_id,
        "category": category
        if category is not None
        else ("person" if owner_id else None),
        "entity_review_status": "trusted" if trusted else "unknown",
        "category_review_status": "trusted" if trusted else "unknown",
        "geometry_review_status": "trusted" if trusted else "unknown",
        "binding_review_status": "trusted" if trusted else "unknown",
        "review_provenance": {
            "source": "synthetic crop review",
            "reviewer": "focused-test",
            "confidence": "high" if trusted else "unresolved",
            "comment": role,
        },
    }


def _trajectory(
    trajectory_id: str,
    image_id: int,
    *,
    duplicate_count: int = 2,
    final_role: str = "accepted",
) -> tuple[
    dict[str, object], list[dict[str, object]], dict[str, object], list[list[int]]
]:
    """Build ``normal A -> normal B -> duplicate B^k -> recovery C -> normal D``."""

    rows = [_row(701, 151670), _row(702, 151680)]
    rows.extend(
        _row(710 + index, 151690 + 10 * index) for index in range(duplicate_count)
    )
    rows.extend([_row(703, 151720), _row(704, 151730)])
    generated = [token for row in rows for token in row]
    prompt = [1, 151655, 151655, 2]
    trajectory = {
        "trajectory_id": trajectory_id,
        "image_id": image_id,
        "split": "train",
        "image": {
            "image_id": image_id,
            "path": f"/synthetic/{image_id}.jpg",
            "content_sha256": "a" * 64,
            "width": 16,
            "height": 16,
        },
        "executed_prompt_token_ids": prompt,
        "executed_prompt_token_ids_sha256": token_ids_sha256(prompt),
        "image_pad_interval": [1, 3],
        "generated_token_ids": generated,
        "generated_token_ids_sha256": token_ids_sha256(generated),
        "generation_provenance": {
            "mode": "sampled",
            "seed": 17,
            "temperature": 0.4,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "checkpoint_id": "b" * 64,
            "prompt_token_ids_sha256": token_ids_sha256(prompt),
        },
    }
    duplicate_start = 2
    recovery_index = duplicate_start + duplicate_count
    reviews = [
        _review(trajectory_id, 0, f"{image_id}:a", "accepted"),
        _review(trajectory_id, 1, f"{image_id}:b", "accepted"),
    ]
    reviews.extend(
        _review(
            trajectory_id,
            duplicate_start + index,
            f"{image_id}:b",
            "duplicate",
            burst_id="burst-b",
        )
        for index in range(duplicate_count)
    )
    reviews.append(_review(trajectory_id, recovery_index, f"{image_id}:c", "recovery"))
    reviews.append(
        _review(trajectory_id, recovery_index + 1, f"{image_id}:d", final_role)
    )
    source = {
        "event_id": f"source-{trajectory_id}-burst-b",
        "trajectory_id": trajectory_id,
        "burst_id": "burst-b",
        "image_id": image_id,
        "opaque_source_payload": {"must": "remain byte-identical across arms"},
    }
    return trajectory, reviews, source, rows


def _assembly(
    *, final_role: str = "accepted"
) -> tuple[dict[str, object], list[list[int]]]:
    trajectory, ledger, source, rows = _trajectory("traj-1", 1, final_role=final_role)
    return (
        build_duplicate_trajectory_bank_drafts(
            rollout_trajectories=[trajectory],
            reviewed_owner_ledger=ledger,
            source_preservation_events=[source],
        ),
        rows,
    )


def _checkpoint_identity() -> dict[str, str]:
    return {
        "adapter_fingerprint": "1" * 64,
        "embedding_delta_fingerprint": "2" * 64,
        "base_config_sha256": "3" * 64,
        "tokenizer_sha256": "4" * 64,
        "token_identity_sha256": "5" * 64,
        "special_token_identity_sha256": "6" * 64,
        "processor_identity_sha256": "7" * 64,
    }


def _attach_materialization_evidence(
    *,
    tmp_path: Path,
    trajectory: dict[str, object],
    source: dict[str, object],
    rows: list[list[int]],
) -> dict[str, str]:
    """Give the synthetic fixture the real image/core evidence a StateBank needs."""

    image_path = tmp_path / "image-1.png"
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(image_path)
    image = {
        "image_id": 1,
        "path": str(image_path),
        "width": 16,
        "height": 16,
        "content_sha256": sha256_file(image_path),
    }
    entities = [
        {
            "entity_id": f"1:{owner}",
            "category": "person",
            "entity_trusted": True,
            "geometry_trusted": True,
            "reference_bbox": [10, 20, 30, 40],
            "review_source": "synthetic crop review",
            "reviewer": "focused-test",
            "review_confidence": "high",
            "comment": "trusted fixture owner",
        }
        for owner in ("a", "b", "c", "d")
    ]
    checkpoint = _checkpoint_identity()
    checkpoint_id = sha256_json(checkpoint)
    trajectory["image"] = image
    trajectory["physical_entities"] = entities
    provenance = trajectory["generation_provenance"]
    assert isinstance(provenance, dict)
    provenance["checkpoint_id"] = checkpoint_id
    prompt = trajectory["executed_prompt_token_ids"]
    assert isinstance(prompt, list)
    source_event_id = str(source["event_id"])
    source_candidate_id = f"{source_event_id}-candidate"
    source_generation = {
        "mode": "greedy",
        "seed": 0,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "checkpoint_id": checkpoint_id,
        "prompt_token_ids_sha256": token_ids_sha256(prompt),
        "prefix_token_ids_sha256": token_ids_sha256([]),
    }
    source["state_bank_rollout"] = {
        "event_id": source_event_id,
        "image": image,
        "split": "train",
        "split_group_id": "image:1",
        "executed_prompt_token_ids": prompt,
        "executed_prompt_token_ids_sha256": token_ids_sha256(prompt),
        "image_pad_interval": [1, 3],
        "prefix_token_ids": [],
        "prefix_token_ids_sha256": token_ids_sha256([]),
        "candidates": [
            {
                "candidate_id": source_candidate_id,
                "token_ids": rows[0],
                "token_ids_sha256": token_ids_sha256(rows[0]),
                "generation_provenance": source_generation,
                "evidence_text": "Source first owner",
            }
        ],
    }
    source["state_bank_review"] = {
        "event_id": source_event_id,
        "admission_status": "accepted",
        "rejection_reason": None,
        "physical_entities": entities,
        "prefix_object_row_count": 0,
        "prefix_coverage_status": "empty",
        "prefix_covered_owner_proofs": [],
        "entity_transition_eligible": False,
        "coordinate_boundary_eligible": False,
        "source_route_imitation_eligible": True,
        "image_balanced_event_weight": 1.0,
        "candidates": [
            {
                "candidate_id": source_candidate_id,
                "role": "positive",
                "harmful_kind": None,
                "physical_owner_id": "1:a",
                "coverage_status": "uncovered",
                "entity_review_status": "trusted",
                "geometry_review_status": "trusted",
                "entity_eligible": True,
                "geometry_eligible": False,
                "owner_resolution_interval": [0, len(rows[0])],
                "coordinate_decision": None,
                "selected_sites": exact_row_site_types(rows[0]),
            }
        ],
        "review_provenance": {"policy": "synthetic Source row"},
    }
    return checkpoint


def _mechanism_events(
    assembly: dict[str, object], bank_name: str
) -> list[dict[str, object]]:
    banks = assembly["banks"]
    assert isinstance(banks, dict)
    bank = banks[bank_name]
    assert isinstance(bank, dict)
    events = bank["events"]
    assert isinstance(events, list)
    return [event for event in events if event["event_family"] != "source_preservation"]


def test_exact_integer_rows_keep_true_generation_and_replay_prefix_hashes() -> None:
    assembly, rows = _assembly()

    local = _mechanism_events(assembly, LOCAL)
    assert len(local) == 2
    assert [event["raw_burst_credit"] for event in local] == [0.5, 0.5]

    # C was originally generated after both B duplicates, but the first local
    # comparison replays it at the first duplicate boundary.  The evidence
    # therefore must keep the two identities distinct.
    first = local[0]
    positive = first["positive_candidate"]
    duplicate = first["duplicate_candidate"]
    assert first["replay_prefix_token_ids"] == rows[0] + rows[1]
    assert positive["token_ids"] == rows[4]
    assert positive["generation_prefix_token_ids_sha256"] == token_ids_sha256(
        rows[0] + rows[1] + rows[2] + rows[3]
    )
    assert first["replay_prefix_token_ids_sha256"] == token_ids_sha256(
        rows[0] + rows[1]
    )
    assert (
        positive["generation_prefix_token_ids_sha256"]
        != first["replay_prefix_token_ids_sha256"]
    )
    assert (
        duplicate["generation_prefix_token_ids_sha256"]
        == first["replay_prefix_token_ids_sha256"]
    )
    evidence = first["duplicate_trajectory_evidence"]
    assert evidence["replay_context_kind"] == "exact_self_prefix_transplant"
    assert evidence["retained_first_owner_row_index"] == 1
    assert evidence["duplicate_row_indices"] == [2, 3]
    assert evidence["removed_row_indices"] == []

    # The second local comparison is at the actually observed D2 prefix, yet
    # the recovery candidate keeps its later, true generation prefix.
    second = local[1]
    assert second["replay_prefix_token_ids"] == rows[0] + rows[1] + rows[2]
    assert (
        second["positive_candidate"]["generation_prefix_token_ids_sha256"]
        == positive["generation_prefix_token_ids_sha256"]
    )


def test_cleaned_suffix_deletes_only_exact_duplicate_rows_and_retains_first_owner() -> (
    None
):
    assembly, rows = _assembly()
    cleaned = _mechanism_events(assembly, CLEANED)
    assert len(cleaned) == 2

    first, second = cleaned
    assert first["positive_candidate"]["token_ids"] == rows[4]
    assert first["replay_prefix_token_ids"] == rows[0] + rows[1]
    assert second["positive_candidate"]["token_ids"] == rows[5]
    assert second["replay_prefix_token_ids"] == rows[0] + rows[1] + rows[4]
    assert first["positive_candidate"][
        "generation_prefix_token_ids_sha256"
    ] == token_ids_sha256(rows[0] + rows[1] + rows[2] + rows[3])
    assert (
        first["duplicate_trajectory_evidence"]["replay_context_kind"]
        == "counterfactual_rewritten"
    )
    assert first["duplicate_trajectory_evidence"]["removed_row_indices"] == [2, 3]
    assert first["retained_first_owner_row_index"] == 1

    receipt = assembly["allocation_receipt"]
    assert isinstance(receipt, dict)
    trajectory_receipt = receipt["trajectories"][0]
    cleaned_trajectory = trajectory_receipt["cleaned_trajectories"]["burst-b"]
    assert cleaned_trajectory["retained_row_indices"] == [0, 1, 4, 5]
    assert (
        cleaned_trajectory["generated_token_ids"]
        == rows[0] + rows[1] + rows[4] + rows[5]
    )


def test_one_credit_per_burst_then_equal_image_balance() -> None:
    first, first_ledger, first_source, _ = _trajectory("traj-1", 1, duplicate_count=2)
    second, second_ledger, second_source, _ = _trajectory(
        "traj-2", 2, duplicate_count=1
    )
    assembly = build_duplicate_trajectory_bank_drafts(
        rollout_trajectories=[first, second],
        reviewed_owner_ledger=[*first_ledger, *second_ledger],
        source_preservation_events=[first_source, second_source],
    )

    receipt = assembly["allocation_receipt"]
    assert isinstance(receipt, dict)
    local_receipt = receipt["arms"][LOCAL]
    assert local_receipt["mechanism_credit_by_burst"] == {
        "traj-1:burst-b": 1.0,
        "traj-2:burst-b": 1.0,
    }
    assert local_receipt["mechanism_credit_by_image_after_balance"] == {
        "1": 1.0,
        "2": 1.0,
    }
    local = _mechanism_events(assembly, LOCAL)
    credits_by_trajectory: dict[str, list[float]] = {}
    for event in local:
        credits_by_trajectory.setdefault(str(event["trajectory_id"]), []).append(
            float(event["raw_burst_credit"])
        )
    assert credits_by_trajectory == {"traj-1": [0.5, 0.5], "traj-2": [1.0]}


def test_unmatched_or_uncertain_rows_are_neutral_and_stop_cleaned_suffix() -> None:
    assembly, rows = _assembly(final_role="unmatched")
    cleaned = _mechanism_events(assembly, CLEANED)
    assert len(cleaned) == 1
    assert cleaned[0]["positive_candidate"]["token_ids"] == rows[4]
    assert all(
        event["positive_candidate"].get("physical_owner_id") != "1:d"
        for event in cleaned
    )
    receipt = assembly["allocation_receipt"]
    assert isinstance(receipt, dict)
    assert {(item["row_index"], item["reason"]) for item in receipt["exclusions"]} >= {
        (5, "neutral_unmatched")
    }


def test_neutral_prefix_rows_are_partially_resolved_with_retained_duplicate_owner_proof(
    tmp_path: Path,
) -> None:
    """An exact neutral prefix row is retained without inventing an owner proof."""

    trajectory, ledger, source, rows = _trajectory(
        "traj-neutral-prefix", 1, duplicate_count=1
    )
    # accepted B -> neutral -> duplicate B -> explicit recovery C
    ledger[0]["physical_owner_id"] = "1:b"
    ledger[1] = _review("traj-neutral-prefix", 1, None, "neutral")
    checkpoint = _attach_materialization_evidence(
        tmp_path=tmp_path, trajectory=trajectory, source=source, rows=rows
    )
    assembly = build_duplicate_trajectory_bank_drafts(
        rollout_trajectories=[trajectory],
        reviewed_owner_ledger=ledger,
        source_preservation_events=[source],
    )

    for bank_name in (LOCAL, POSITIVE_ONLY, CLEANED):
        event = _mechanism_events(assembly, bank_name)[0]
        context = event["state_bank_context"]
        assert event["replay_prefix_token_ids"] == rows[0] + rows[1]
        assert context["prefix_object_row_count"] == 2
        assert context["prefix_coverage_status"] == "partially_resolved"
        assert [
            (proof["prefix_object_row_index"], proof["owner_id"])
            for proof in context["prefix_covered_owner_proofs"]
        ] == [(0, "1:b")]

    receipt = materialize_duplicate_trajectory_state_banks(
        output_dir=tmp_path / "partial-prefix-materialized",
        assembly=assembly,
        source_checkpoint=checkpoint,
        prompt_identity_sha256="e" * 64,
        source_artifacts=[{"artifact_id": "synthetic", "sha256": "f" * 64}],
    )
    assert receipt["status"] == "materialized"
    assert receipt["arms"][LOCAL]["state_bank_validation_receipt"]["record_count"] == 2


def test_matched_arms_keep_source_dose_and_family_stratified_optimizer_windows() -> (
    None
):
    assembly, _ = _assembly()
    banks = assembly["banks"]
    assert isinstance(banks, dict)
    assert tuple(banks) == BANK_NAMES

    source_events_by_arm = {
        name: [
            event
            for event in banks[name]["events"]
            if event["event_family"] == "source_preservation"
        ]
        for name in BANK_NAMES
    }
    assert all(
        events == source_events_by_arm[SOURCE_ONLY]
        for events in source_events_by_arm.values()
    )
    assert len(source_events_by_arm[SOURCE_ONLY]) == 1

    # Positive-only and local use byte-identical recovery candidates and replay
    # prefixes; the only treatment difference is the duplicate candidate.
    positives = _mechanism_events(assembly, POSITIVE_ONLY)
    locals_ = _mechanism_events(assembly, LOCAL)
    assert [event["replay_prefix_token_ids_sha256"] for event in positives] == [
        event["replay_prefix_token_ids_sha256"] for event in locals_
    ]
    assert [event["positive_candidate"] for event in positives] == [
        event["positive_candidate"] for event in locals_
    ]
    assert all("duplicate_candidate" not in event for event in positives)
    assert all("duplicate_candidate" in event for event in locals_)

    combined = _mechanism_events(assembly, COMBINED)
    by_family = {}
    for event in combined:
        by_family.setdefault(event["event_family"], 0.0)
        by_family[event["event_family"]] += float(
            event["mechanism_credit_before_image_balance"]
        )
    assert by_family == {
        "local_duplicate_rejection": pytest.approx(0.5),
        "duplicate_cleaned_imitation": pytest.approx(0.5),
    }
    assert all(len(banks[name]["optimizer_windows"]) == 1 for name in BANK_NAMES)
    combined_window = banks[COMBINED]["optimizer_windows"][0]
    assert set(combined_window["global_family_denominators"]) == {
        "source_preservation",
        "local_duplicate_rejection",
        "duplicate_cleaned_imitation",
    }
    assert all(
        value > 0 for value in combined_window["global_family_denominators"].values()
    )


def test_source_weighting_groups_by_immutable_source_image_and_overrides_review_dose() -> None:
    first, first_ledger, first_source, _ = _trajectory(
        "traj-source-1", 1, duplicate_count=1
    )
    second, second_ledger, second_source, _ = _trajectory(
        "traj-source-2", 2, duplicate_count=1
    )
    third, third_ledger, third_source, _ = _trajectory(
        "traj-source-3", 3, duplicate_count=1
    )
    for source, donor_image_id in (
        (first_source, "90"),
        (second_source, "90"),
        (third_source, "91"),
    ):
        source["immutable_source_image_id"] = donor_image_id
        source["state_bank_rollout"] = {"event_id": source["event_id"]}
        source["state_bank_review"] = {
            "event_id": source["event_id"],
            "image_balanced_event_weight": 99.0,
        }

    assembly = build_duplicate_trajectory_bank_drafts(
        rollout_trajectories=[first, second, third],
        reviewed_owner_ledger=[*first_ledger, *second_ledger, *third_ledger],
        source_preservation_events=[first_source, second_source, third_source],
    )
    source_events = assembly["banks"][SOURCE_ONLY]["events"]
    assert [
        (
            event["immutable_source_image_id"],
            event["source_balance_image_id"],
            event["image_source_event_count"],
            event["image_balanced_event_weight"],
        )
        for event in source_events
    ] == [
        ("90", "90", 2, 0.5),
        ("90", "90", 2, 0.5),
        ("91", "91", 1, 1.0),
    ]
    assert all(
        event["state_bank_review"]["image_balanced_event_weight"] == 99.0
        for event in source_events
    )

    rollouts, reviews = state_bank_rows_from_duplicate_draft_bank(
        assembly["banks"][SOURCE_ONLY]
    )
    assert [item["event_id"] for item in rollouts] == [
        event["event_id"] for event in source_events
    ]
    assert [item["image_balanced_event_weight"] for item in reviews] == [
        0.5,
        0.5,
        1.0,
    ]


def test_writer_is_immutable_and_emits_one_allocation_receipt(tmp_path: Path) -> None:
    assembly, _ = _assembly()
    output = tmp_path / "duplicate-banks"
    write_duplicate_trajectory_bank_drafts(output, assembly)
    assert (output / "allocation-receipt.json").is_file()
    for name in BANK_NAMES:
        assert (output / name / "pre-state-bank" / "events.jsonl").is_file()
        assert (output / name / "optimizer-windows.json").is_file()
    with pytest.raises(AssemblyError, match="already exists"):
        write_duplicate_trajectory_bank_drafts(output, assembly)


def test_materializes_all_five_banks_against_typed_duplicate_state_bank_contract(
    tmp_path: Path,
) -> None:
    trajectory, ledger, source, rows = _trajectory("traj-1", 1)
    checkpoint = _attach_materialization_evidence(
        tmp_path=tmp_path, trajectory=trajectory, source=source, rows=rows
    )
    assembly = build_duplicate_trajectory_bank_drafts(
        rollout_trajectories=[trajectory],
        reviewed_owner_ledger=ledger,
        source_preservation_events=[source],
    )

    receipt = materialize_duplicate_trajectory_state_banks(
        output_dir=tmp_path / "materialized",
        assembly=assembly,
        source_checkpoint=checkpoint,
        prompt_identity_sha256="e" * 64,
        source_artifacts=[{"artifact_id": "synthetic", "sha256": "f" * 64}],
    )

    assert receipt["status"] == "materialized"
    for name in BANK_NAMES:
        arm = receipt["arms"][name]
        assert arm["rollout_row_count"] == arm["review_row_count"]
        assert (
            tmp_path / "materialized" / name / "state-bank" / "manifest.json"
        ).is_file()
    assert receipt["arms"][SOURCE_ONLY]["rollout_row_count"] == 1
    assert receipt["arms"][LOCAL]["rollout_row_count"] == 3
    assert receipt["arms"][CLEANED]["rollout_row_count"] == 3
    assert receipt["arms"][COMBINED]["rollout_row_count"] == 5


def test_exact_row_slicing_refuses_retokenization_or_partial_rows() -> None:
    first = _row(900, 151670)
    second = _row(901, 151680)
    prefix, candidate = exact_row_slices(first + second, 1)
    assert prefix == first
    assert candidate == second
    with pytest.raises(AssemblyError, match="partial or non-row"):
        exact_row_slices(first + second + [1234], 1)
