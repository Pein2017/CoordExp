from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest

from src.inference.parsing import parse_compact_object_box_closed
from src.label_studio_coco_refinement.categories import (
    COCO80_CATEGORIES,
    COCO80_REGISTRY,
    Coco80Registry,
    CocoCategory,
)
from src.label_studio_coco_refinement.inference_results import (
    CategoryIdentity,
    CurrentTarget,
    InferenceAttemptReceipt,
    InferenceResultContractError,
    Outcome,
    RequestLifecycle,
    RequestState,
    RequestTarget,
    bind_for_insertion,
    classify_parser_result,
    coco80_category_resolver,
    finalize_region_links,
    terminal_state_for_result,
)
from src.label_studio_coco_refinement.roi_transform import RoiLetterboxTransform
from src.templates.renderer import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    OBJECT_REF_END_TOKEN,
    OBJECT_REF_START_TOKEN,
)


def _object(name: str, coords: tuple[int, int, int, int]) -> str:
    coord_text = "".join(f"<|coord_{value}|>" for value in coords)
    return (
        f"{OBJECT_REF_START_TOKEN}{name}{OBJECT_REF_END_TOKEN}"
        f"{BOX_START_TOKEN}{coord_text}{BOX_END_TOKEN}"
    )


def _transform(*, wide: bool = False) -> RoiLetterboxTransform:
    return RoiLetterboxTransform.from_label_studio_roi(
        source_width=100,
        source_height=50 if wide else 100,
        roi=(0, 0, 100, 100),
        canvas_width=100,
        canvas_height=100,
    )


def _target(transform: RoiLetterboxTransform) -> RequestTarget:
    return RequestTarget(
        request_id="request-1",
        project_id="train",
        task_id="task-7",
        task_epoch="epoch-3",
        image_id="image-42",
        annotation_id="annotation-9",
        annotation_revision="rev-11",
        profile_fingerprint="a" * 64,
        project_generation=8,
        transform_fingerprint=transform.fingerprint,
        preexisting_draft_dirty=True,
    )


def _profile_receipt(target: RequestTarget) -> dict[str, object]:
    return {
        "schema_version": "coordexp-roi-engine-profile-v1",
        "profile_name": "test-profile",
        "profile_fingerprint": target.profile_fingerprint,
        "endpoint": "http://127.0.0.1:8123/infer",
        "artifacts": [
            {
                "role": role,
                "path": f"/test/{role}",
                "kind": "file",
                "sha256": chr(ord("b") + index) * 64,
                "file_count": 1,
                "total_bytes": 1,
            }
            for index, role in enumerate(
                ("base_weights", "model_config", "processor", "tokenizer")
            )
        ],
        "identity_fingerprints": {
            field: "f" * 64
            for field in (
                "resolved_config",
                "prompt_policy",
                "parser",
                "adapter",
                "transform",
                "transformers",
                "processor_kwargs",
                "runtime",
            )
        },
        "processor": {
            "factor": 32,
            "default_canvas": [1024, 1024],
            "axis_bounds": [32, 2048],
            "max_total_pixels": 2_097_152,
            "do_resize": False,
        },
        "deadline_seconds": 20.0,
    }


def _classify(text: str, *, wide: bool = False):
    transform = _transform(wide=wide)
    target = _target(transform)
    parsed = parse_compact_object_box_closed(
        text,
        row_id=target.request_id,
        row_index=0,
        image_width=transform.canvas_width,
        image_height=transform.canvas_height,
    )
    return classify_parser_result(
        parse_row=parsed,
        raw_response_text=text,
        target=target,
        transform=transform,
    )


def _finalized_accepted_result():
    result = _classify(_object("person", (100, 100, 500, 500)))
    return finalize_region_links(
        result,
        CurrentTarget(**result.target.binding_payload()),
        region_links={"request-1:result-0": "region-1"},
    )


def _terminal_receipt(result, *, wide: bool = False):
    lifecycle = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
    lifecycle = lifecycle.transition(terminal_state_for_result(result), at_seconds=2)
    return InferenceAttemptReceipt(
        target=result.target,
        lifecycle=lifecycle,
        profile_receipt=_profile_receipt(result.target),
        transform_receipt=_transform(wide=wide).to_receipt_dict(),
        result=result,
    )


def test_accepted_result_is_one_target_bound_append_payload_with_raw_replay() -> None:
    raw = _object("person", (100, 200, 700, 900))
    result = _classify(raw)

    assert result.outcome is Outcome.ACCEPTED
    assert result.clear_roi is True
    assert (result.parsed_count, result.inserted_count, result.rejected_count) == (1, 1, 0)
    assert result.insertion_payload is not None
    payload = result.insertion_payload.to_dict()
    assert payload["mode"] == "append_one_undo_action"
    assert len(payload["regions"]) == 1
    assert payload["regions"][0]["category_id"] == 1
    receipt = result.to_receipt_dict()
    assert receipt["raw_response_text"] == raw
    assert receipt["raw_response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert receipt["results"][0]["raw_span_text"] == raw
    assert receipt["results"][0]["coord_bins"] == [100, 200, 700, 900]
    assert receipt["results"][0]["inverse_mapping"] is not None


def test_unknown_class_is_valid_parse_but_all_rejected_and_clears_roi() -> None:
    result = _classify(_object("spaceship", (100, 100, 500, 500)))

    assert result.outcome is Outcome.ALL_REJECTED
    assert result.clear_roi is True
    assert result.insertion_payload is None
    assert result.records[0].reject_reason == "unsupported_or_noncanonical_coco80_class"


def test_shared_registry_adapter_returns_official_sparse_stop_sign_id() -> None:
    resolve = coco80_category_resolver(COCO80_REGISTRY)

    assert resolve("stop sign") == CategoryIdentity("stop sign", 13)
    assert resolve("Traffic Light") is None


def test_counterfeit_contiguous_coco80_registry_is_rejected() -> None:
    contiguous = Coco80Registry(
        tuple(
            CocoCategory(index, category.name)
            for index, category in enumerate(COCO80_CATEGORIES, start=1)
        )
    )

    with pytest.raises(InferenceResultContractError, match="canonical sparse"):
        coco80_category_resolver(contiguous)


def test_mixed_valid_and_class_rejected_is_accepted_with_drops() -> None:
    result = _classify(
        _object("person", (100, 100, 400, 400))
        + _object("Person", (500, 500, 900, 900))
    )

    assert result.outcome is Outcome.ACCEPTED_WITH_DROPS
    assert result.inserted_count == 1
    assert result.rejected_count == 1
    assert result.insertion_payload is not None
    assert len(result.insertion_payload.regions) == 1


def test_parser_drops_are_retained_alongside_valid_result() -> None:
    result = _classify(_object("person", (100, 100, 400, 400)) + " trailing")

    assert result.parser_status == "accepted_with_drops"
    assert result.outcome is Outcome.ACCEPTED_WITH_DROPS
    assert any(record.reject_reason == "parser:unmatched_text" for record in result.records)


@pytest.mark.parametrize(
    ("raw", "outcome", "clear_roi"),
    [
        ("", Outcome.EMPTY, True),
        ("not the compact grammar", Outcome.ALL_SPANS_DROPPED, False),
        ('[{"class": "person"}]', Outcome.UNSUPPORTED_FORMAT, False),
    ],
)
def test_empty_and_response_failure_outcome_table(
    raw: str, outcome: Outcome, clear_roi: bool
) -> None:
    result = _classify(raw)
    assert result.outcome is outcome
    assert result.clear_roi is clear_roi
    assert result.insertion_payload is None


@pytest.mark.parametrize(
    "raw",
    ["", "not the compact grammar", '[{"class": "person"}]'],
)
def test_terminal_receipt_full_replay_preserves_noninserting_outcomes(raw: str) -> None:
    result = _classify(raw)

    receipt = _terminal_receipt(result)

    assert receipt.result == result


def test_parser_valid_box_entirely_in_padding_becomes_all_rejected() -> None:
    result = _classify(_object("person", (0, 0, 100, 100)), wide=True)

    assert result.outcome is Outcome.ALL_REJECTED
    assert result.records[0].reject_reason == "mapped_entirely_in_padding"
    assert result.insertion_payload is None


def test_terminal_receipt_full_replay_preserves_mapping_rejection() -> None:
    result = _classify(_object("person", (0, 0, 100, 100)), wide=True)

    receipt = _terminal_receipt(result, wide=True)

    assert receipt.result == result


def test_stale_annotation_binding_abandons_without_returning_mutation_payload() -> None:
    result = _classify(_object("person", (100, 100, 500, 500)))
    target = result.target
    current = CurrentTarget(
        project_id=target.project_id,
        task_id=target.task_id,
        task_epoch=target.task_epoch,
        image_id=target.image_id,
        annotation_id=target.annotation_id,
        annotation_revision="new-revision",
        profile_fingerprint=target.profile_fingerprint,
        project_generation=target.project_generation,
    )

    decision = bind_for_insertion(result, current)

    assert decision.status == "abandoned_before_insertion"
    assert decision.payload is None
    assert decision.mismatches == ("annotation_revision",)


def test_exact_target_binding_preserves_atomic_payload() -> None:
    result = _classify(_object("person", (100, 100, 500, 500)))
    target = result.target
    current = CurrentTarget(**target.binding_payload())

    decision = bind_for_insertion(result, current)

    assert decision.status == "bound"
    assert decision.payload == result.insertion_payload


def test_region_links_finalize_only_after_exact_binding() -> None:
    result = _classify(_object("person", (100, 100, 500, 500)))
    current = CurrentTarget(**result.target.binding_payload())

    finalized = finalize_region_links(
        result,
        current,
        region_links={"request-1:result-0": "region-stable-17"},
    )

    assert finalized.insertion_payload is not None
    assert finalized.insertion_payload.regions[0].region_link == "region-stable-17"
    assert finalized.records[0].region_link == "region-stable-17"


def test_region_link_finalization_rejects_stale_target_or_missing_links() -> None:
    result = _classify(_object("person", (100, 100, 500, 500)))
    target = result.target
    stale = CurrentTarget(
        **{**target.binding_payload(), "annotation_revision": "stale"}
    )
    with pytest.raises(InferenceResultContractError, match="abandoned"):
        finalize_region_links(
            result,
            stale,
            region_links={"request-1:result-0": "region-stable-17"},
        )
    with pytest.raises(InferenceResultContractError, match="exactly match"):
        finalize_region_links(
            result,
            CurrentTarget(**target.binding_payload()),
            region_links={},
        )


def test_lifecycle_requires_explicit_cooperative_cancellation_terminal() -> None:
    lifecycle = RequestLifecycle()
    lifecycle = lifecycle.transition(RequestState.RUNNING, at_seconds=1)
    lifecycle = lifecycle.transition(
        RequestState.CANCELLING, at_seconds=2, reason="deadline reached"
    )
    lifecycle = lifecycle.transition(
        RequestState.TIMEOUT_FAILURE,
        at_seconds=3,
        reason="backend acknowledged stop",
    )

    assert lifecycle.state is RequestState.TIMEOUT_FAILURE
    assert lifecycle.state.terminal is True
    with pytest.raises(InferenceResultContractError, match="terminal"):
        lifecycle.transition(RequestState.RUNTIME_FAILURE, at_seconds=4, reason="late")


def test_lifecycle_rejects_skipped_or_reasonless_cancellation() -> None:
    lifecycle = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
    with pytest.raises(InferenceResultContractError, match="invalid request transition"):
        lifecycle.transition(RequestState.CANCELLED, at_seconds=2, reason="stop")
    with pytest.raises(InferenceResultContractError, match="explicit reason"):
        lifecycle.transition(RequestState.CANCELLING, at_seconds=2)


def test_failure_receipt_records_stage_and_never_claims_annotation_mutation() -> None:
    transform = _transform()
    target = _target(transform)
    lifecycle = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
    lifecycle = lifecycle.transition(
        RequestState.RUNTIME_FAILURE, at_seconds=2, reason="generation failed"
    )
    receipt = InferenceAttemptReceipt(
        target=target,
        lifecycle=lifecycle,
        profile_receipt=_profile_receipt(target),
        transform_receipt=transform.to_receipt_dict(),
        result=None,
        failure_stage="generation",
        failure_code="backend_runtime_error",
        failure_message="generation failed",
    )

    payload = receipt.to_dict()
    assert payload["terminal_status"] == "runtime_failure"
    assert payload["failure"]["annotation_mutated"] is False
    assert "message" not in payload["failure"]


def test_accepted_receipt_requires_and_retains_finalized_region_links() -> None:
    result = _classify(_object("stop sign", (100, 100, 500, 500)))
    current = CurrentTarget(**result.target.binding_payload())
    finalized = finalize_region_links(
        result,
        current,
        region_links={"request-1:result-0": "region-13"},
    )
    lifecycle = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
    lifecycle = lifecycle.transition(RequestState.ACCEPTED, at_seconds=2)

    receipt = InferenceAttemptReceipt(
        target=result.target,
        lifecycle=lifecycle,
        profile_receipt=_profile_receipt(result.target),
        transform_receipt=_transform().to_receipt_dict(),
        result=finalized,
    )

    payload = receipt.to_dict()
    region = payload["result"]["insertion_payload"]["regions"][0]
    assert region["category_id"] == 13
    assert region["region_link"] == "region-13"


def test_attempt_receipt_rejects_forged_class_and_insertion_category() -> None:
    finalized = _finalized_accepted_result()
    assert finalized.insertion_payload is not None
    forged_record = replace(
        finalized.records[0],
        canonical_category_name="bicycle",
        official_category_id=2,
    )
    forged_region = replace(
        finalized.insertion_payload.regions[0],
        category_name="bicycle",
        category_id=2,
    )
    forged = replace(
        finalized,
        records=(forged_record,),
        insertion_payload=replace(
            finalized.insertion_payload,
            regions=(forged_region,),
        ),
    )

    with pytest.raises(InferenceResultContractError, match="classification semantics"):
        _terminal_receipt(forged)


def test_attempt_receipt_rejects_forged_class_decision_and_reject_reason() -> None:
    finalized = _finalized_accepted_result()
    forged_record = replace(
        finalized.records[0],
        class_decision="rejected",
        reject_reason="forged_reclassification",
    )
    forged = replace(finalized, records=(forged_record,))

    with pytest.raises(InferenceResultContractError, match="classification semantics"):
        _terminal_receipt(forged)


def test_attempt_receipt_rejects_forged_bbox_with_inverse_removed() -> None:
    finalized = _finalized_accepted_result()
    assert finalized.insertion_payload is not None
    forged_bbox = (10, 20, 30, 40)
    forged_record = replace(
        finalized.records[0],
        inverse_mapping=None,
        final_norm1000_bbox=forged_bbox,
    )
    forged_region = replace(
        finalized.insertion_payload.regions[0],
        norm1000_bbox=forged_bbox,
    )
    forged = replace(
        finalized,
        records=(forged_record,),
        insertion_payload=replace(
            finalized.insertion_payload,
            regions=(forged_region,),
        ),
    )

    with pytest.raises(InferenceResultContractError, match="classification semantics"):
        _terminal_receipt(forged)


def test_attempt_receipt_rejects_forged_rejected_count() -> None:
    forged = replace(_finalized_accepted_result(), rejected_count=999)

    with pytest.raises(InferenceResultContractError, match="classification semantics"):
        _terminal_receipt(forged)


def test_attempt_receipt_rejects_forged_outcome_and_matching_lifecycle() -> None:
    forged = replace(
        _finalized_accepted_result(),
        outcome=Outcome.ACCEPTED_WITH_DROPS,
        rejected_count=1,
    )

    with pytest.raises(InferenceResultContractError, match="classification semantics"):
        _terminal_receipt(forged)


def test_attempt_receipt_rejects_result_rebound_to_different_raw_response() -> None:
    result = _classify(_object("person", (100, 100, 500, 500)))
    finalized = finalize_region_links(
        result,
        CurrentTarget(**result.target.binding_payload()),
        region_links={"request-1:result-0": "region-1"},
    )
    replacement_raw = _object("bicycle", (100, 100, 500, 500))
    rebound = replace(
        finalized,
        raw_response_text=replacement_raw,
        raw_response_sha256=hashlib.sha256(replacement_raw.encode()).hexdigest(),
    )
    lifecycle = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
    lifecycle = lifecycle.transition(RequestState.ACCEPTED, at_seconds=2)

    with pytest.raises(InferenceResultContractError, match="parse row hash"):
        InferenceAttemptReceipt(
            target=result.target,
            lifecycle=lifecycle,
            profile_receipt=_profile_receipt(result.target),
            transform_receipt=_transform().to_receipt_dict(),
            result=rebound,
        )


@pytest.mark.parametrize(
    ("record_changes", "error_match"),
    [
        ({"char_start": 1}, "raw span offsets/content"),
        ({"char_end": 1}, "raw span offsets/content"),
        ({"raw_span_text": "tampered"}, "raw span offsets/content"),
        ({"raw_span_sha256": "0" * 64}, "raw span hash"),
        ({"parsed_description": "bicycle"}, "replay records"),
    ],
)
def test_attempt_receipt_revalidates_each_record_against_raw_response(
    record_changes: dict[str, object],
    error_match: str,
) -> None:
    result = _classify(_object("person", (100, 100, 500, 500)))
    finalized = finalize_region_links(
        result,
        CurrentTarget(**result.target.binding_payload()),
        region_links={"request-1:result-0": "region-1"},
    )
    altered_record = replace(finalized.records[0], **record_changes)
    altered = replace(finalized, records=(altered_record,))
    lifecycle = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
    lifecycle = lifecycle.transition(RequestState.ACCEPTED, at_seconds=2)

    with pytest.raises(InferenceResultContractError, match=error_match):
        InferenceAttemptReceipt(
            target=result.target,
            lifecycle=lifecycle,
            profile_receipt=_profile_receipt(result.target),
            transform_receipt=_transform().to_receipt_dict(),
            result=altered,
        )


def test_attempt_receipt_revalidates_parse_row_hash_against_raw_response() -> None:
    result = _classify(_object("person", (100, 100, 500, 500)))
    finalized = finalize_region_links(
        result,
        CurrentTarget(**result.target.binding_payload()),
        region_links={"request-1:result-0": "region-1"},
    )
    altered = replace(finalized, parse_row_sha256="0" * 64)
    lifecycle = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
    lifecycle = lifecycle.transition(RequestState.ACCEPTED, at_seconds=2)

    with pytest.raises(InferenceResultContractError, match="parse row hash"):
        InferenceAttemptReceipt(
            target=result.target,
            lifecycle=lifecycle,
            profile_receipt=_profile_receipt(result.target),
            transform_receipt=_transform().to_receipt_dict(),
            result=altered,
        )


def test_accepted_receipt_rejects_missing_region_links() -> None:
    result = _classify(_object("person", (100, 100, 500, 500)))
    lifecycle = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
    lifecycle = lifecycle.transition(RequestState.ACCEPTED, at_seconds=2)

    with pytest.raises(InferenceResultContractError, match="finalized non-null"):
        InferenceAttemptReceipt(
            target=result.target,
            lifecycle=lifecycle,
            profile_receipt=_profile_receipt(result.target),
            transform_receipt=_transform().to_receipt_dict(),
            result=result,
        )


def test_failure_cancel_and_abandon_receipts_forbid_results() -> None:
    result = _classify(_object("person", (100, 100, 500, 500)))
    for state in (
        RequestState.RUNTIME_FAILURE,
        RequestState.ABANDONED_BEFORE_INSERTION,
    ):
        lifecycle = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
        lifecycle = lifecycle.transition(state, at_seconds=2, reason="terminal")
        with pytest.raises(InferenceResultContractError, match="cannot contain a result"):
            InferenceAttemptReceipt(
                target=result.target,
                lifecycle=lifecycle,
                profile_receipt=_profile_receipt(result.target),
                transform_receipt=_transform().to_receipt_dict(),
                result=result,
                failure_stage="runtime",
                failure_code="terminal",
            )

    cancelling = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
    cancelling = cancelling.transition(
        RequestState.CANCELLING, at_seconds=2, reason="user"
    )
    cancelled = cancelling.transition(
        RequestState.CANCELLED, at_seconds=3, reason="backend_stopped"
    )
    with pytest.raises(InferenceResultContractError, match="cannot contain a result"):
        InferenceAttemptReceipt(
            target=result.target,
            lifecycle=cancelled,
            profile_receipt=_profile_receipt(result.target),
            transform_receipt=_transform().to_receipt_dict(),
            result=result,
            failure_stage="generation",
            failure_code="cancelled",
        )


def test_result_terminal_receipt_forbids_every_failure_field() -> None:
    result = _classify("")
    lifecycle = RequestLifecycle().transition(RequestState.RUNNING, at_seconds=1)
    lifecycle = lifecycle.transition(RequestState.EMPTY, at_seconds=2)

    with pytest.raises(InferenceResultContractError, match="cannot contain failure"):
        InferenceAttemptReceipt(
            target=result.target,
            lifecycle=lifecycle,
            profile_receipt=_profile_receipt(result.target),
            transform_receipt=_transform().to_receipt_dict(),
            result=result,
            failure_message="contradiction",
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"headers": {"Authorization": "Bearer secret"}},
        {"Cookie": "session=secret"},
        {"private_key": "-----BEGIN PRIVATE KEY-----"},
    ],
)
def test_attempt_receipt_rejects_non_allowlisted_profile_credentials(
    extra: dict[str, object],
) -> None:
    transform = _transform()
    target = _target(transform)
    lifecycle = RequestLifecycle().transition(
        RequestState.PROFILE_FAILURE,
        at_seconds=1,
        reason="profile_invalid",
    )
    with pytest.raises(InferenceResultContractError, match="non-allowlisted"):
        InferenceAttemptReceipt(
            target=target,
            lifecycle=lifecycle,
            profile_receipt={**_profile_receipt(target), **extra},
            transform_receipt=transform.to_receipt_dict(),
            result=None,
            failure_stage="profile",
            failure_code="profile_invalid",
        )


def test_result_terminal_state_classification_is_explicit() -> None:
    accepted = _classify(_object("person", (100, 100, 500, 500)))
    response_failure = _classify("bad response")
    assert terminal_state_for_result(accepted) is RequestState.ACCEPTED
    assert terminal_state_for_result(response_failure) is RequestState.RESPONSE_FAILURE


def test_parse_row_cannot_be_reused_with_unrelated_raw_response() -> None:
    transform = _transform()
    target = _target(transform)
    original = _object("person", (100, 100, 500, 500))
    parsed = parse_compact_object_box_closed(
        original,
        row_id=target.request_id,
        row_index=0,
        image_width=100,
        image_height=100,
    )

    with pytest.raises(InferenceResultContractError, match="exact current-parser"):
        classify_parser_result(
            parse_row=parsed,
            raw_response_text=_object("stop sign", (100, 100, 500, 500)),
            target=target,
            transform=transform,
        )


def test_parse_row_altered_span_offsets_are_rejected() -> None:
    transform = _transform()
    target = _target(transform)
    raw = _object("person", (100, 100, 500, 500))
    parsed = parse_compact_object_box_closed(
        raw,
        row_id=target.request_id,
        row_index=0,
        image_width=100,
        image_height=100,
    )
    predictions = deepcopy(parsed.predictions)
    predictions[0]["char_start"] = 1
    altered = replace(parsed, predictions=predictions)

    with pytest.raises(InferenceResultContractError, match="exact current-parser"):
        classify_parser_result(
            parse_row=altered,
            raw_response_text=raw,
            target=target,
            transform=transform,
        )


def test_parser_and_request_identity_mismatch_is_rejected() -> None:
    transform = _transform()
    target = _target(transform)
    parsed = parse_compact_object_box_closed(
        _object("person", (100, 100, 500, 500)),
        row_id="wrong-request",
        row_index=0,
        image_width=100,
        image_height=100,
    )
    with pytest.raises(InferenceResultContractError, match="row identity"):
        classify_parser_result(
            parse_row=parsed,
            raw_response_text="ignored",
            target=target,
            transform=transform,
        )
