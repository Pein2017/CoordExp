from __future__ import annotations

from pathlib import Path

import pytest

from src.common.errors import PackingContractError
from src.qwen.forward import build_qwen_forward_inputs
from src.rollout_calibration import (
    assemble_state_bank,
    build_exact_replay_segment,
    load_state_bank,
    plan_calibration_micro_steps,
)
from src.rollout_calibration.state_bank import STATE_BANK_MANIFEST_NAME
from conftest import (
    IMAGE_TOKEN_ID,
    source_artifacts,
    synthetic_inputs,
)


def _loaded_bank(tmp_path, checkpoint_identity, prompt_identity_sha256):
    rollouts, reviews, _ = synthetic_inputs(tmp_path)
    assemble_state_bank(
        output_dir=tmp_path / "bank",
        rollout_rows=rollouts,
        review_rows=reviews,
        source_checkpoint=checkpoint_identity,
        prompt_identity_sha256=prompt_identity_sha256,
        source_artifacts=source_artifacts(),
    )
    return load_state_bank(
        tmp_path / "bank" / STATE_BANK_MANIFEST_NAME,
        expected_source_checkpoint=checkpoint_identity,
        expected_prompt_identity_sha256=prompt_identity_sha256,
    )


def test_exact_replay_uses_stored_prompt_prefix_and_candidate_without_tokenizer(
    tmp_path: Path,
    checkpoint_identity,
    prompt_identity_sha256: str,
    fake_components,
    processor_config,
) -> None:
    bank = _loaded_bank(tmp_path, checkpoint_identity, prompt_identity_sha256)
    event = bank.records[0]
    candidate = event.candidates[0]

    replay = build_exact_replay_segment(
        event,
        candidate,
        components=fake_components,
        processor_config=processor_config,
        global_max_length=128,
    )

    assert replay.input_ids == (
        *event.executed_prompt_token_ids,
        *event.prefix_token_ids,
        *candidate.token_ids,
    )
    assert replay.candidate_token_start == len(event.executed_prompt_token_ids) + len(
        event.prefix_token_ids
    )
    assert replay.image_pad_physical_start == 1
    assert replay.image_pad_physical_end == 5


def test_atomic_planning_keeps_candidates_isolated_and_maps_selected_positions(
    tmp_path: Path,
    checkpoint_identity,
    prompt_identity_sha256: str,
    fake_components,
    processor_config,
) -> None:
    bank = _loaded_bank(tmp_path, checkpoint_identity, prompt_identity_sha256)

    micro_step = plan_calibration_micro_steps(
        bank,
        split="train",
        components=fake_components,
        processor_config=processor_config,
        global_max_length=128,
        image_token_id=IMAGE_TOKEN_ID,
    )[0]

    assert len(micro_step.pack.segments) == len(bank.records[0].candidates) == 2
    assert [segment.example_id for segment in micro_step.pack.segments] == [
        replay.example_id for replay in micro_step.replay_segments
    ]
    assert micro_step.position_inputs.reset_points == tuple(
        segment.start for segment in micro_step.pack.segments
    )
    metadata = micro_step.calibration_metadata
    assert metadata.event_id == bank.records[0].event_id
    assert len(metadata.candidates) == 2
    positive = next(
        candidate for candidate in metadata.candidates if candidate.role == "positive"
    )
    source_positive = next(
        candidate
        for candidate in bank.records[0].candidates
        if candidate.role == "positive"
    )
    assert positive.entity_review_status == source_positive.entity_review_status
    assert positive.geometry_review_status == source_positive.geometry_review_status
    positive_replay = micro_step.replay_segments[positive.segment_index]
    positive_segment = micro_step.pack.segments[positive.segment_index]
    expected_first_target = (
        positive_segment.start + positive_replay.candidate_token_start
    )
    assert positive.owner_resolution_physical_target_interval == (
        expected_first_target,
        expected_first_target + 1,
    )
    assert positive.selected_sites[0].physical_target_position == expected_first_target
    assert (
        positive.selected_sites[0].physical_logits_position == expected_first_target - 1
    )
    assert set(metadata.selected_logits_positions) == {
        site.physical_logits_position
        for candidate in metadata.candidates
        for site in candidate.selected_sites
    }
    assert (
        metadata.selected_causal_logits_positions is metadata.selected_logits_positions
    )

    forward_inputs = build_qwen_forward_inputs(
        micro_step.pack,
        micro_step.replay_segments,
        micro_step.position_inputs,
        logits_to_keep_positions=metadata.selected_logits_positions,
    )
    assert forward_inputs.logits_position_ids == metadata.selected_logits_positions
    assert forward_inputs.input_ids[0].tolist() == list(micro_step.pack.input_ids)
    assert fake_components.processor.image_processor.batch_sizes == [2]


def test_atomic_event_capacity_fails_without_partial_pack(
    tmp_path: Path,
    checkpoint_identity,
    prompt_identity_sha256: str,
    fake_components,
    processor_config,
) -> None:
    bank = _loaded_bank(tmp_path, checkpoint_identity, prompt_identity_sha256)
    # Each individual segment has length 11 or 9, but the full event needs 20.
    with pytest.raises(PackingContractError) as exc_info:
        plan_calibration_micro_steps(
            bank,
            split="train",
            components=fake_components,
            processor_config=processor_config,
            global_max_length=12,
            image_token_id=IMAGE_TOKEN_ID,
        )
    assert exc_info.value.code == "rollout_calibration.event_capacity"
    assert exc_info.value.context["event_id"] == "synthetic-event-1"
    assert exc_info.value.context["required_length"] == 20
