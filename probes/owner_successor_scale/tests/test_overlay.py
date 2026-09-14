import copy
import json

import pytest

from probes.owner_successor_scale.overlay import (
    CANDIDATE_SCHEMA,
    NATIVE_CONVENTION,
    REVIEW_SCHEMA,
    OverlayError,
    append_review_event,
    build_candidate_manifest,
    canonical_json,
    export_positive_patch,
    export_training_patch,
    sha256_bytes,
    training_export_is_safe,
    validate_candidate,
)


def _binding(name: str) -> dict:
    return {"path": f"/tmp/owner-overlay-{name}", "sha256": "0" * 64, "size_bytes": 1}


def _candidate(
    number: int,
    *,
    image_id: int = 417044,
    source_split: str = "train",
    train_exposed: bool = True,
    eval_exposed: bool = False,
    confirmation: bool = False,
    group: bool = False,
) -> dict:
    width, height = 1152, 864
    x1 = 20 + number * 70
    bbox = [x1, 100, x1 + 40, 140]
    return {
        "schema": CANDIDATE_SCHEMA,
        "proposal_id": f"C{number:02d}",
        "image": {
            "image_id": image_id,
            "example_id": f"coco2017_{source_split}_{image_id:012d}",
            "path": f"/data/images/{image_id}.jpg",
            "sha256": "1" * 64,
            "size_bytes": 10,
            "width": width,
            "height": height,
            "split": source_split,
        },
        "proposal": {
            "native_bbox": bbox,
            "native_coordinate_convention": NATIVE_CONVENTION,
            "coord_bins": [
                round(1000 * bbox[0] / width),
                round(1000 * bbox[1] / height),
                round(1000 * bbox[2] / width),
                round(1000 * bbox[3] / height),
            ],
            "bin_coordinate_convention": "norm1000_xyxy",
            "proposal_kind": "group" if group else "singleton",
            "owner_granularity": "group" if group else "singleton",
            "scope": "group_extent" if group else "singleton",
            "class_proposal": "donut",
            "source_prediction_index": number,
        },
        "provenance": {
            "native_source_manifest": _binding(f"source-{number}"),
            "native_gt_manifest": _binding(f"gt-{number}"),
            "candidate_checkpoint": _binding(f"checkpoint-{number}"),
            "candidate_output": {
                "binding": _binding(f"output-{number}"),
                "row_id": f"row-{number}",
                "row_sha256": "2" * 64,
            },
        },
        "review_evidence": {
            "render_route": "view_image",
            "cards": [_binding(f"card-{number}")],
        },
        "roles": {
            "source_split": source_split,
            "train_exposed": train_exposed,
            "eval_exposed": eval_exposed,
            "confirmation": confirmation,
            "panel_role": "exposed_train_review_candidate" if source_split == "train" else "evaluation_panel",
        },
        "status": "candidate",
        "training_target": False,
    }


def _event(
    candidate: dict,
    *,
    seq: int = 0,
    status: str = "lead_reviewed_positive",
    owner_id: str | None = "owner-1",
    reviewer_role: str = "lead",
    lead_accepted: bool = True,
    owner_granularity: str = "singleton",
    scope: str = "singleton",
    alias: str | None = None,
    parent: str | None = None,
    children: list[str] | None = None,
) -> dict:
    return {
        "schema": REVIEW_SCHEMA,
        "review_id": f"review-{candidate['proposal_id']}-{seq}",
        "proposal_id": candidate["proposal_id"],
        "review_seq": seq,
        "reviewer_id": "root" if reviewer_role == "lead" else "worker",
        "reviewer_role": reviewer_role,
        "reviewer_route": "view_image",
        "reviewed_at": "2026-09-13T12:00:00Z",
        "status": status,
        "lead_accepted": lead_accepted,
        "owner_id": owner_id,
        "same_image_alias_of": alias,
        "parent_proposal_id": parent,
        "child_proposal_ids": children or [],
        "owner_granularity": owner_granularity,
        "scope": scope,
        "instance_entity": "resolved" if status != "candidate" else "unknown",
        "geometry": "group_extent" if status == "group_extent" else ("resolved" if status != "candidate" else "unknown"),
        "class": "resolved" if status != "candidate" else "unknown",
        "evidence": [_binding(f"review-{candidate['proposal_id']}-{seq}")],
    }


def _selection(image_ids: list[int], excluded_image_ids: list[int]) -> dict:
    return {
        "schema": "native_owner_successor_scale_throughput.confirmation_selection.v1",
        "status": "frozen_cpu_no_model_calls",
        "image_ids": image_ids,
        "excluded_image_ids": excluded_image_ids,
        "image_ids_sha256": sha256_bytes(canonical_json(image_ids).encode("utf-8")),
    }


def test_unknown_and_worker_review_stay_neutral(tmp_path):
    candidate = _candidate(1)
    worker = _event(
        candidate,
        status="candidate",
        owner_id=None,
        reviewer_role="worker",
        lead_accepted=False,
        owner_granularity="unknown",
        scope="unknown",
    )
    payload = export_positive_patch([candidate], [worker], tmp_path / "positive.json")
    assert payload["records"] == []
    assert payload["negative_count"] == 0
    assert payload["unknown_is_neutral"] is True


def test_duplicate_same_image_aliases_are_rejected(tmp_path):
    first, second = _candidate(1), _candidate(2)
    first_event = _event(first, owner_id="same-owner")
    second_event = _event(second, owner_id="same-owner")
    with pytest.raises(OverlayError, match="duplicate same-image owner/alias"):
        export_positive_patch([first, second], [first_event, second_event], tmp_path / "positive.json")


def test_distinct_owner_ids_with_explicit_alias_fail_closed(tmp_path):
    first, second = _candidate(1), _candidate(2)
    first_event = _event(first, owner_id="canonical-owner")
    second_event = _event(second, owner_id="different-owner", alias=first["proposal_id"])
    with pytest.raises(OverlayError, match="same-image alias cannot export"):
        export_positive_patch([first, second], [first_event, second_event], tmp_path / "positive.json")


def test_resolved_non_coco80_class_cannot_export_as_positive(tmp_path):
    candidate = _candidate(1)
    candidate["proposal"]["class_proposal"] = "dvd"
    with pytest.raises(OverlayError, match="outside canonical COCO-80"):
        export_positive_patch([candidate], [_event(candidate)], tmp_path / "positive.json")


def test_confirmation_image_cannot_leak_but_train_excluded_id_is_explicitly_allowed():
    candidate = _candidate(1)
    event = _event(candidate)
    admission = {"status": "lead_accepted", "path": "/root/owns/admission.json"}
    selection = _selection([999], [417044])
    assert training_export_is_safe([candidate], [event], admission=admission, confirmation_selection=selection)

    leaked = copy.deepcopy(candidate)
    leaked["roles"]["eval_exposed"] = True
    leaked["roles"]["confirmation"] = True
    selection = _selection([417044], [417044])
    assert not training_export_is_safe([leaked], [_event(leaked)], admission=admission, confirmation_selection=selection)
    assert not training_export_is_safe([candidate], [event], admission=admission)


def test_training_export_requires_frozen_confirmation_selection(tmp_path):
    candidate = _candidate(1)
    with pytest.raises(OverlayError, match="frozen confirmation selection"):
        export_training_patch(
            [candidate],
            [_event(candidate)],
            tmp_path / "training.json",
            admission={"status": "lead_accepted"},
        )


def test_group_extent_is_preserved_as_evidence_but_not_training_target(tmp_path):
    candidate = _candidate(1, group=True)
    event = _event(
        candidate,
        status="group_extent",
        owner_id="group-owner",
        owner_granularity="group",
        scope="group_extent",
    )
    payload = export_positive_patch([candidate], [event], tmp_path / "positive.json")
    assert payload["group_extent_count"] == 1
    assert payload["records"][0]["owner_granularity"] == "group"
    with pytest.raises(OverlayError, match="group extent"):
        export_training_patch(
            [candidate],
            [event],
            tmp_path / "training.json",
            admission={"status": "lead_accepted"},
            confirmation_selection=_selection([], []),
        )


def test_candidate_table_is_small_and_source_blind_card_manifest():
    candidates = [_candidate(i) for i in range(1, 9)]
    manifest = build_candidate_manifest(candidates)
    assert manifest["candidate_count"] == 8
    assert manifest["source_blind_cards"] is True
    assert manifest["status"] == "candidate_review_pending"
    assert all(item["status"] == "candidate" for item in manifest["records"])


def test_append_review_is_append_only_and_rejects_duplicate_ids(tmp_path):
    candidate = _candidate(1)
    path = tmp_path / "reviews.jsonl"
    event = _event(candidate, status="candidate", owner_id=None, reviewer_role="worker", lead_accepted=False, owner_granularity="unknown", scope="unknown")
    append_review_event(path, event)
    with pytest.raises(OverlayError, match="duplicate review_id"):
        append_review_event(path, event)
    assert len(path.read_text().splitlines()) == 1
    assert json.loads(path.read_text())["proposal_id"] == candidate["proposal_id"]


def test_review_sequence_cannot_move_backward(tmp_path):
    candidate = _candidate(1)
    path = tmp_path / "reviews.jsonl"
    first = _event(candidate, seq=1, status="candidate", owner_id=None, reviewer_role="worker", lead_accepted=False, owner_granularity="unknown", scope="unknown")
    second = _event(candidate, seq=0, status="candidate", owner_id=None, reviewer_role="worker", lead_accepted=False, owner_granularity="unknown", scope="unknown")
    append_review_event(path, first)
    with pytest.raises(OverlayError, match="monotonically"):
        append_review_event(path, second)


def test_pixel_bin_drift_is_fail_closed():
    candidate = _candidate(1)
    candidate["proposal"]["coord_bins"][0] += 20
    with pytest.raises(OverlayError, match="pixel/binned box identity drift"):
        validate_candidate(candidate)
