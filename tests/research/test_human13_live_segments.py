from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts.research.build_human13_k_union_manifest import (
    DuplicateEventRecord,
    PredictionRowInput,
    PrefixRecord,
    RequestIdentity,
    SelectedRowRecord,
    TrajectoryRecord,
)
from scripts.research.human13_live_segments import materialize_segments
from scripts.research import run_human13_k_union_overfit as runner


@dataclass(frozen=True)
class Skeleton:
    example_id: str
    input_ids: tuple[int, ...]
    prompt_token_count: int
    owner_row_tokens: dict[str, tuple[int, ...]]


def _request(seed=None, mode="source_greedy"):
    return RequestIdentity(
        "hf" if seed is None else "vllm", "test", mode, 1, seed, 0, 0.0, 1.0, 1.0, 3084
    )


def _manifest():
    source = TrajectoryRecord(
        "source",
        _request(),
        (10, 11, 12, 99),
        3,
        "im_end",
        "complete",
        (),
        PrefixRecord((10, 11, 12), (10, 11, 12), ()),
        (),
        (),
        (),
        (True, True, True),
        (False, False, False, False),
    )
    sampled = TrajectoryRecord(
        "k-21001",
        RequestIdentity("vllm", "test", "k_sampled", 1, 21001, 0, 0.4, 0.95, 1.1, 512),
        (20, 21, 22, 23, 99),
        4,
        "im_end",
        "complete",
        (
            PredictionRowInput(
                "k-row", 0, "object", (0.1, 0.1, 0.2, 0.2), 1, 3, 2
            ),
        ),
        PrefixRecord((20, 21, 22, 23), (20, 21, 22, 23), ()),
        ("k-row",),
        (),
        ("k-row",),
        (False,) * 5,
        (False,) * 5,
    )
    selected = SelectedRowRecord(
        "gt:1:0", "k-row", "k-21001", 21001, 0, 1.0, (21, 22), (True, True)
    )
    event = DuplicateEventRecord(
        "dup:1:k-21001:r", 1, "k-21001", "r", "keep", (20, 21), 22, 2
    )
    owner = type("O", (), {"owner_id": "gt:1:0", "stratum": "H"})()
    return type(
        "M",
        (),
        {
            "images": (
                type(
                    "I",
                    (),
                    {
                        "image_id": 1,
                        "trajectories": (source, sampled),
                        "owners": (owner,),
                        "selected_rows": (selected,),
                        "duplicate_events": (event,),
                        "replay_row_ids": (),
                        "target_row_ids": (),
                        "candidate_row_ids": ("k-row",),
                    },
                )(),
            )
        },
    )()


def test_materializes_roles_from_manifest_spans_without_retokenizing():
    skeleton = Skeleton("image:1", (7, 8, 9), 3, {"gt:1:0": (30, 31)})
    result = materialize_segments(
        _manifest(), {1: skeleton}, prompt_token_counts={1: 3}
    )
    by_role = {}
    for segment in result.segments:
        by_role.setdefault(segment.role, []).append(segment)
    assert by_role["a1_full_h"][0].encoded_example.input_ids == (
        7,
        8,
        9,
        10,
        11,
        12,
        21,
        22,
    )
    assert by_role["h1_independent"][0].encoded_example.input_ids == (
        7,
        8,
        9,
        10,
        11,
        12,
        21,
        22,
    )
    assert by_role["duplicate_event"][0].encoded_example.input_ids == (
        7,
        8,
        9,
        20,
        21,
        22,
    )
    assert by_role["full_gt"][0].encoded_example.input_ids == (7, 8, 9, 30, 31)
    assert by_role["source_replay"][0].encoded_example.input_ids == (
        7,
        8,
        9,
        10,
        11,
        12,
    )
    h_binding = by_role["h1_independent"][0].encoded_example.human13_row_bindings[0]
    assert h_binding.unit_id == "gt:1:0" and h_binding.target_token_mask == (True, True)
    dup_binding = by_role["duplicate_event"][0].encoded_example.human13_row_bindings[0]
    assert dup_binding.family == "duplicate" and dup_binding.target_token_mask == (
        False,
        False,
        True,
    )
    gt_binding = by_role["full_gt"][0].encoded_example.human13_row_bindings[0]
    assert gt_binding.family == "full_gt" and gt_binding.unit_id == "gt:1:0"


def test_a4_atomic_preflight_reports_aggregate_and_rejects_over_limit():
    skeleton = Skeleton("image:1", (1, 2), 2, {"gt:1:0": (30,)})
    result = materialize_segments(
        _manifest(),
        {1: skeleton},
        prompt_token_counts={1: 2},
        global_max_length=8,
    )
    with pytest.raises(ValueError, match="A4.*12,000"):
        result.preflight(3, enforce_a4_aggregate=True)


def test_a4_aggregate_does_not_block_non_a4_materialization():
    skeleton = Skeleton("image:1", (1, 2), 2, {"gt:1:0": (30,)})
    result = materialize_segments(
        _manifest(),
        {1: skeleton},
        prompt_token_counts={1: 2},
        global_max_length=8,
    )
    result.preflight(8)


def test_image_without_h_owners_does_not_emit_empty_h_segments():
    manifest = _manifest()
    image = manifest.images[0]
    source = image.trajectories[0]
    g_owner = SimpleNamespace(
        owner_id="gt:1:0",
        stratum="G",
        source_row_ids=(),
        sampled_row_ids=(),
    )
    no_h_image = SimpleNamespace(
        image_id=1,
        trajectories=(source,),
        owners=(g_owner,),
        selected_rows=(),
        duplicate_events=(),
        replay_row_ids=(),
        target_row_ids=(),
        candidate_row_ids=(),
    )
    no_h_manifest = SimpleNamespace(images=(no_h_image,))
    skeleton = Skeleton("image:1", (7, 8, 9), 3, {"gt:1:0": (30, 31)})

    result = materialize_segments(no_h_manifest, {1: skeleton})

    roles = {segment.role for segment in result.segments}
    assert "a1_full_h" not in roles
    assert "a8_full_h" not in roles
    assert "h1_independent" not in roles
    assert "a6_donor_h1" not in roles


def test_materialized_a4_candidates_are_independent_runner_segments():
    skeleton = Skeleton("image:1", (7, 8, 9), 3, {"gt:1:0": (30, 31)})
    result = materialize_segments(
        _manifest(), {1: skeleton}, prompt_token_counts={1: 3}
    )
    a4 = [item for item in result.segments if item.role == "a4_union"]
    assert len(a4) == 1
    assert runner.build_logical_segments(a4)[0].role == "a4_union"


def test_a6_donor_prefix_removes_earlier_duplicate_row_tokens():
    manifest = _manifest()
    image = manifest.images[0]
    source, sampled = image.trajectories
    duplicate = PredictionRowInput(
        "dup-row", 0, "object", (0.0, 0.0, 0.1, 0.1), 0, 2, 1
    )
    retained = PredictionRowInput(
        "keep-row", 1, "object", (0.2, 0.2, 0.3, 0.3), 2, 3, 2
    )
    target = PredictionRowInput(
        "k-row", 2, "object", (0.4, 0.4, 0.5, 0.5), 3, 5, 4
    )
    sampled_with_duplicate = TrajectoryRecord(
        sampled.trajectory_id,
        sampled.request,
        (40, 41, 42, 21, 22, 99),
        5,
        sampled.stop_reason,
        sampled.parser_status,
        (duplicate, retained, target),
        PrefixRecord((40, 41, 42, 21, 22), (42, 21, 22), ("dup-row",)),
        ("keep-row", "k-row"),
        ("dup-row",),
        ("k-row",),
        (False,) * 6,
        (False,) * 6,
    )
    manifest_with_duplicate = SimpleNamespace(
        images=(
            SimpleNamespace(
                image_id=image.image_id,
                trajectories=(source, sampled_with_duplicate),
                owners=image.owners,
                selected_rows=image.selected_rows,
                duplicate_events=image.duplicate_events,
                replay_row_ids=image.replay_row_ids,
                target_row_ids=image.target_row_ids,
                candidate_row_ids=image.candidate_row_ids,
            ),
        )
    )
    skeleton = Skeleton("image:1", (7, 8, 9), 3, {"gt:1:0": (30, 31)})

    result = materialize_segments(manifest_with_duplicate, {1: skeleton})

    donor = next(item for item in result.segments if item.role == "a6_donor_h1")
    assert donor.encoded_example.input_ids == (7, 8, 9, 42, 21, 22)
    binding = donor.encoded_example.human13_row_bindings[0]
    assert (binding.token_start, binding.token_end) == (4, 6)


def test_source_replay_binding_is_offset_after_multimodal_prompt():
    source = TrajectoryRecord(
        "source",
        _request(),
        (10, 11, 12, 99),
        3,
        "im_end",
        "complete",
        (
            type(
                "R",
                (),
                {"row_id": "source-row", "token_start": 0, "token_end": 3},
            )(),
        ),
        PrefixRecord((10, 11, 12), (10, 11, 12), ()),
        ("source-row",),
        (),
        (),
        (True, True, True),
        (False, False, False, False),
    )
    owner = type(
        "O",
        (),
        {
            "owner_id": "gt:1:0",
            "stratum": "G",
            "source_row_ids": ("source-row",),
            "sampled_row_ids": (),
        },
    )()
    image = type(
        "I",
        (),
        {
            "image_id": 1,
            "trajectories": (source,),
            "owners": (owner,),
            "selected_rows": (),
            "duplicate_events": (),
            "replay_row_ids": ("source-row",),
            "target_row_ids": (),
            "candidate_row_ids": (),
        },
    )()
    manifest = type("M", (), {"images": (image,)})()
    skeleton = Skeleton("image:1", (7, 8, 9), 3, {"gt:1:0": (30,)})

    result = materialize_segments(manifest, {1: skeleton})
    replay = next(item for item in result.segments if item.role == "source_replay")
    binding = replay.encoded_example.human13_row_bindings[0]
    assert replay.encoded_example.input_ids[
        binding.token_start : binding.token_end
    ] == (
        10,
        11,
        12,
    )
    assert (binding.token_start, binding.token_end) == (3, 6)
