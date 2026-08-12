from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

import src.packing.planner as planner_module
from src.common.errors import PackingContractError
from src.coordinate_targets import CoordinateLossTarget
from src.packing.planner import (
    DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET,
    ONLINE_WINDOW_BINPACK,
    SOURCE_ORDER_NEXT_FIT,
    WINDOW_BINPACK,
    PackPlan,
    PackPlanCursor,
    PackPlanStreamReceipt,
    create_pack_plan,
    plan_packed_sequences,
    replay_pack_plan,
)
from src.qwen.encoding import EncodedExample, EncodedTokenSpan
from src.qwen.images import (
    QwenExecutedImageEvidence,
    QwenImageEncoding,
    QwenNoResizeImagePlan,
)


def test_legacy_wrapper_artifact_bytes_and_behavior_remain_compatible() -> None:
    examples = _examples((4, 3, 5))

    packs = plan_packed_sequences(examples, global_max_length=7)
    payload = json.dumps(
        [pack.to_artifact_dict() for pack in packs],
        sort_keys=True,
        separators=(",", ":"),
    )

    assert hashlib.sha256(payload.encode()).hexdigest() == (
        "02365cf013166552047fdbe0509d5a2db278d7779a4a504575c07344e6c7fe86"
    )
    plan = create_pack_plan(
        examples,
        global_max_length=7,
        policy=SOURCE_ORDER_NEXT_FIT,
    )
    assert replay_pack_plan(plan, examples) == packs


def test_window_binpack_preserves_atomic_input_and_intra_image_order_identity() -> None:
    examples = _examples((6, 6, 4, 4))

    plan = create_pack_plan(
        examples,
        global_max_length=10,
        policy=WINDOW_BINPACK,
        window_size=4,
        seed=71,
    )
    replayed = replay_pack_plan(plan, examples)

    assert [pack.input_ordinals for pack in plan.packs] == [(1, 2), (0, 3)]
    assert [item.intra_image_order_identity for item in plan.inputs] == [
        example.intra_image_order_identity for example in examples
    ]
    for pack in replayed:
        for segment in pack.segments:
            assert (
                pack.input_ids[segment.start : segment.end]
                == examples[segment.example_index].input_ids
            )


@pytest.mark.parametrize(
    ("policy", "kwargs"),
    [
        (SOURCE_ORDER_NEXT_FIT, {}),
        (WINDOW_BINPACK, {"window_size": 4}),
        (ONLINE_WINDOW_BINPACK, {"lookahead": 3, "max_packs": 100}),
    ],
)
def test_every_input_has_exactly_one_packed_or_rejected_disposition(
    policy: str,
    kwargs: dict[str, int],
) -> None:
    examples = _examples((4, 0, 11, 6, 3))

    plan = create_pack_plan(
        examples,
        global_max_length=10,
        policy=policy,
        seed=19,
        **kwargs,
    )

    packed = [ordinal for pack in plan.packs for ordinal in pack.input_ordinals]
    rejected = [item.input_ordinal for item in plan.rejected_examples]
    assert sorted(packed + rejected) == list(range(len(examples)))
    assert len(packed + rejected) == len(set(packed + rejected))
    assert {item.reason for item in plan.rejected_examples} == {
        "encoded_example_empty",
        "encoded_example_exceeds_global_max_length",
    }


@pytest.mark.parametrize(
    ("policy", "kwargs"),
    [
        (SOURCE_ORDER_NEXT_FIT, {}),
        (WINDOW_BINPACK, {"window_size": 5}),
        (ONLINE_WINDOW_BINPACK, {"lookahead": 4, "max_packs": 100}),
    ],
)
def test_worker_count_remains_semantic_pending_upstream_materialization_equality(
    policy: str,
    kwargs: dict[str, int],
) -> None:
    examples = _examples((7, 3, 6, 4, 2, 8))

    one_worker = create_pack_plan(
        examples,
        global_max_length=10,
        policy=policy,
        seed=123,
        worker_count=1,
        **kwargs,
    )
    eight_workers = create_pack_plan(
        examples,
        global_max_length=10,
        policy=policy,
        seed=123,
        worker_count=8,
        **kwargs,
    )

    assert one_worker.requested_worker_count == 1
    assert eight_workers.requested_worker_count == 8
    assert one_worker.semantic_identity_sha256 != eight_workers.semantic_identity_sha256
    assert one_worker.worker_count_disposition == (
        "semantic_pending_upstream_materialization_equality"
    )
    assert one_worker.worker_count_integration_requirement == (
        "canonical_concurrent_encoded_materialization_and_complete_plan_equality"
    )


def test_public_policy_identity_exactly_matches_plan_parameters() -> None:
    identity = planner_module.build_pack_plan_policy_identity(
        policy=ONLINE_WINDOW_BINPACK,
        lookahead=7,
        seed=23,
        worker_count=8,
        cursor_byte_budget=8_192,
        fragment_item_budget=96,
        fragment_byte_budget=65_536,
    )

    assert identity == {
        "policy": "online_window_binpack",
        "algorithm_version": "coordexp-swift-online-window-binpack-v3",
        "window_size": None,
        "lookahead": 7,
        "tie_breaker": "source_anchor_best_fit_seeded_identity_then_ordinal_v1",
        "seed": 23,
        "requested_worker_count": 8,
        "worker_count_disposition": (
            "semantic_pending_upstream_materialization_equality"
        ),
        "worker_count_integration_requirement": (
            "canonical_concurrent_encoded_materialization_and_complete_plan_equality"
        ),
        "cursor_byte_budget": 8_192,
        "fragment_item_budget": 96,
        "fragment_byte_budget": 65_536,
    }
    one_worker_identity = planner_module.build_pack_plan_policy_identity(
        policy=ONLINE_WINDOW_BINPACK,
        lookahead=7,
        seed=23,
        worker_count=1,
        cursor_byte_budget=8_192,
        fragment_item_budget=96,
        fragment_byte_budget=65_536,
    )
    assert one_worker_identity["requested_worker_count"] == 1
    assert one_worker_identity != identity

    with pytest.raises(PackingContractError) as exc_info:
        planner_module.build_pack_plan_policy_identity(
            policy=WINDOW_BINPACK,
            window_size=4,
            lookahead=7,
        )
    assert exc_info.value.code == "packing.pack_plan_parameters"


def test_online_plan_cursor_roundtrip_resumes_exact_uninterrupted_sequence() -> None:
    examples = _examples((6, 2, 5, 4, 3, 8, 1, 7, 2))
    expected_memberships: list[tuple[int, ...]] = []
    reference = planner_module.stream_online_pack_plan_fragments(
        lambda: iter(examples),
        fragment_sink=lambda fragment: expected_memberships.extend(
            pack.input_ordinals for pack in fragment.packs
        ),
        global_max_length=10,
        lookahead=3,
        max_packs_per_fragment=100,
        seed=2026,
    )

    cursor: PackPlanCursor | None = None
    predecessor: PackPlan | None = None
    resumed_memberships: list[tuple[int, ...]] = []
    while cursor is None or not cursor.complete:
        prior_cursor = cursor
        part = create_pack_plan(
            examples,
            global_max_length=10,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=3,
            seed=2026,
            replay_cursor=cursor,
            replay_plan=predecessor,
            max_packs=1,
        )
        serialized = part.to_json()
        part = PackPlan.from_json(serialized)
        resumed_memberships.extend(pack.input_ordinals for pack in part.packs)
        assert len(part.replay_cursor.pending_inputs) <= 3
        assert part.max_pending_items_observed <= 3
        if prior_cursor is not None:
            assert part.source_prefix_start_sha256 == prior_cursor.source_prefix_sha256
            assert (
                part.emitted_prefix_start_sha256 == prior_cursor.emitted_prefix_sha256
            )
        cursor = PackPlanCursor.from_dict(part.replay_cursor.to_dict())
        predecessor = part

    assert resumed_memberships == expected_memberships
    assert cursor.source_prefix_sha256 == reference.terminal_cursor.source_prefix_sha256
    assert (
        cursor.emitted_prefix_sha256 == reference.terminal_cursor.emitted_prefix_sha256
    )


def test_online_planner_rejects_an_unbounded_complete_receipt() -> None:
    with pytest.raises(PackingContractError) as exc_info:
        create_pack_plan(
            _examples((1, 1, 1)),
            global_max_length=2,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=2,
            max_packs=None,
        )

    assert exc_info.value.code == "packing.pack_plan_fragment_bound"


def test_online_fragment_obeys_explicit_item_and_byte_budgets() -> None:
    fragment = create_pack_plan(
        _examples((1,) * 100),
        global_max_length=4,
        policy=ONLINE_WINDOW_BINPACK,
        lookahead=4,
        max_packs=100,
        fragment_item_budget=9,
        fragment_byte_budget=16_384,
    )

    assert len(fragment.inputs) <= 9
    assert len(fragment.to_json().encode("ascii")) <= 16_384
    assert fragment.fragment_item_budget == 9
    assert fragment.fragment_byte_budget == 16_384
    assert not fragment.replay_cursor.complete


def test_non_resumable_and_source_drift_cursors_fail_closed() -> None:
    examples = _examples((6, 2, 5, 4, 3))
    part = create_pack_plan(
        examples,
        global_max_length=10,
        policy=ONLINE_WINDOW_BINPACK,
        lookahead=3,
        max_packs=1,
    )

    with pytest.raises(PackingContractError) as exc_info:
        create_pack_plan(
            examples,
            global_max_length=10,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=3,
            replay_cursor=replace(part.replay_cursor, resumable=False),
            replay_plan=part,
            max_packs=1,
        )
    assert exc_info.value.code == "packing.pack_plan_cursor_predecessor"

    with pytest.raises(PackingContractError) as exc_info:
        create_pack_plan(
            examples,
            global_max_length=10,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=4,
            replay_cursor=part.replay_cursor,
            replay_plan=part,
            max_packs=1,
        )
    assert exc_info.value.code == "packing.pack_plan_cursor_policy"

    changed = list(examples)
    pending = part.replay_cursor.pending_inputs[0]
    changed[pending.input_ordinal] = replace(
        changed[pending.input_ordinal],
        row_ids=("changed-row",),
    )
    with pytest.raises(PackingContractError) as exc_info:
        create_pack_plan(
            changed,
            global_max_length=10,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=3,
            replay_cursor=part.replay_cursor,
            replay_plan=part,
            max_packs=1,
        )
    assert exc_info.value.code == "packing.pack_plan_cursor_identity"


def test_online_resume_requires_exact_authenticated_predecessor_plan() -> None:
    examples = _examples((6, 2, 5, 4, 3))
    part = create_pack_plan(
        examples,
        global_max_length=10,
        policy=ONLINE_WINDOW_BINPACK,
        lookahead=3,
        max_packs=1,
    )

    with pytest.raises(PackingContractError) as exc_info:
        create_pack_plan(
            examples,
            global_max_length=10,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=3,
            replay_cursor=part.replay_cursor,
        )
    assert exc_info.value.code == "packing.pack_plan_cursor_predecessor"

    resumed = create_pack_plan(
        examples,
        global_max_length=10,
        policy=ONLINE_WINDOW_BINPACK,
        lookahead=3,
        replay_cursor=part.replay_cursor,
        replay_plan=part,
        max_packs=1,
    )
    assert resumed.predecessor_plan_sha256 == part.canonical_sha256

    forged = replace(part.replay_cursor, emitted_prefix_sha256="0" * 64)
    with pytest.raises(PackingContractError) as exc_info:
        create_pack_plan(
            examples,
            global_max_length=10,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=3,
            replay_cursor=forged,
            replay_plan=part,
        )
    assert exc_info.value.code == "packing.pack_plan_cursor_predecessor"


def test_serialization_is_canonical_and_rejects_corrupt_state() -> None:
    plan = create_pack_plan(
        _examples((6, 6, 4, 4)),
        global_max_length=10,
        policy=WINDOW_BINPACK,
        window_size=4,
        seed=5,
    )

    serialized = plan.to_json()
    assert PackPlan.from_json(serialized) == plan
    assert PackPlan.from_json(serialized).to_json() == serialized

    unknown_field = plan.to_dict()
    unknown_field["unexpected"] = True
    with pytest.raises(PackingContractError) as exc_info:
        PackPlan.from_dict(unknown_field)
    assert exc_info.value.code == "packing.pack_plan_fields"

    corrupt_metric = plan.to_dict()
    corrupt_metric["metrics"]["utilization"] = 0.25
    with pytest.raises(PackingContractError) as exc_info:
        PackPlan.from_dict(_reauthenticate(corrupt_metric))
    assert exc_info.value.code == "packing.pack_plan_metric"

    corrupt_membership = plan.to_dict()
    corrupt_membership["packs"][0]["input_ordinals"][0] = 999
    with pytest.raises(PackingContractError) as exc_info:
        PackPlan.from_dict(_reauthenticate(corrupt_membership))
    assert exc_info.value.code == "packing.pack_plan_membership"

    bool_metric = plan.to_dict()
    bool_metric["metrics"]["pack_count"] = True
    with pytest.raises(PackingContractError) as exc_info:
        PackPlan.from_dict(_reauthenticate(bool_metric))
    assert exc_info.value.code == "packing.pack_plan_field_type"


def test_plan_and_cursor_canonical_sha256_reject_mutation() -> None:
    plan = create_pack_plan(
        _examples((6, 4, 3)),
        global_max_length=10,
        policy=ONLINE_WINDOW_BINPACK,
        lookahead=2,
        max_packs=2,
    )

    assert plan.to_dict()["authentication"]["sha256"] == plan.canonical_sha256
    assert (
        plan.replay_cursor.to_dict()["authentication"]["sha256"]
        == plan.replay_cursor.canonical_sha256
    )
    assert PackPlanCursor.from_json(plan.replay_cursor.to_json()) == plan.replay_cursor

    corrupt_plan = plan.to_dict()
    corrupt_plan["authentication"]["sha256"] = "0" * 64
    with pytest.raises(PackingContractError) as exc_info:
        PackPlan.from_dict(corrupt_plan)
    assert exc_info.value.code == "packing.pack_plan_authentication"

    corrupt_cursor = plan.replay_cursor.to_dict()
    corrupt_cursor["authentication"]["sha256"] = "f" * 64
    with pytest.raises(PackingContractError) as exc_info:
        PackPlanCursor.from_dict(corrupt_cursor)
    assert exc_info.value.code == "packing.pack_plan_authentication"


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    plan = create_pack_plan(
        _examples((6, 4)),
        global_max_length=10,
        policy=SOURCE_ORDER_NEXT_FIT,
    )
    duplicate_key = plan.to_json().replace(
        '{"algorithm_version":',
        '{"algorithm_version":"duplicate","algorithm_version":',
        1,
    )
    with pytest.raises(PackingContractError) as exc_info:
        PackPlan.from_json(duplicate_key)
    assert exc_info.value.code == "packing.pack_plan_json"
    assert "duplicate JSON object key" in str(exc_info.value.cause)

    nonfinite = plan.to_json().replace('"utilization":1.0', '"utilization":NaN', 1)
    with pytest.raises(PackingContractError) as exc_info:
        PackPlan.from_json(nonfinite)
    assert exc_info.value.code == "packing.pack_plan_json"
    assert "non-finite JSON constant" in str(exc_info.value.cause)


def test_complete_plan_rejects_missing_prefix_and_replay_binds_input_ids() -> None:
    examples = _examples((0, 5, 5))
    plan = create_pack_plan(
        examples,
        global_max_length=10,
        policy=SOURCE_ORDER_NEXT_FIT,
    )

    with pytest.raises(PackingContractError) as exc_info:
        replace(
            plan,
            inputs=plan.inputs[1:],
            rejected_examples=(),
        )
    assert exc_info.value.code == "packing.pack_plan_prefix_coverage"

    changed = list(examples)
    changed[1] = replace(changed[1], input_ids=(999, 998, 997, 996, 995))
    with pytest.raises(PackingContractError) as exc_info:
        replay_pack_plan(plan, changed)
    assert exc_info.value.code == "packing.pack_plan_replay_identity"


def test_resume_rejects_missing_or_changed_full_source_prefix() -> None:
    examples = _examples((6, 2, 5, 4, 3, 7))
    part = create_pack_plan(
        examples,
        global_max_length=10,
        policy=ONLINE_WINDOW_BINPACK,
        lookahead=3,
        max_packs=1,
    )
    assert part.replay_cursor.next_input_ordinal > 1

    changed = list(examples)
    changed[0] = replace(
        changed[0],
        input_ids=tuple(reversed(changed[0].input_ids)),
    )
    with pytest.raises(PackingContractError) as exc_info:
        create_pack_plan(
            changed,
            global_max_length=10,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=3,
            replay_cursor=part.replay_cursor,
            replay_plan=part,
            max_packs=1,
        )
    assert exc_info.value.code == "packing.pack_plan_cursor_prefix_identity"

    truncated = examples[: part.replay_cursor.next_input_ordinal - 1]
    with pytest.raises(PackingContractError) as exc_info:
        create_pack_plan(
            truncated,
            global_max_length=10,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=3,
            replay_cursor=part.replay_cursor,
            replay_plan=part,
            max_packs=1,
        )
    assert exc_info.value.code == "packing.pack_plan_cursor_source"


def test_online_cursor_count_and_serialized_byte_budget_stay_bounded() -> None:
    examples = _examples((1,) * 2_000)
    cursor: PackPlanCursor | None = None
    predecessor: PackPlan | None = None
    emitted_count = 0
    byte_budget = 4_096
    while cursor is None or not cursor.complete:
        fragment = create_pack_plan(
            examples,
            global_max_length=1,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=8,
            cursor_byte_budget=byte_budget,
            replay_cursor=cursor,
            replay_plan=predecessor,
            max_packs=100,
        )
        emitted_count += fragment.accepted_example_count
        cursor = PackPlanCursor.from_json(fragment.replay_cursor.to_json())
        predecessor = fragment
        assert len(cursor.pending_inputs) <= 8
        assert cursor.serialized_size_bytes <= byte_budget
        assert fragment.max_serialized_cursor_bytes_observed <= byte_budget
        assert "inputs" not in cursor.to_dict()
        assert "packs" not in cursor.to_dict()

    assert emitted_count == len(examples)
    assert byte_budget < DEFAULT_PACK_PLAN_CURSOR_BYTE_BUDGET


@pytest.mark.parametrize("example_count", [2_000, 20_000])
def test_online_fragment_sink_stays_bounded_and_stitches_exact_prefix(
    example_count: int,
) -> None:
    examples = _examples((1,) * example_count)

    narrow_sink = _FragmentSink(example_count)
    narrow = planner_module.stream_online_pack_plan_fragments(
        lambda: iter(examples),
        fragment_sink=narrow_sink,
        global_max_length=4,
        lookahead=8,
        max_packs_per_fragment=200,
        fragment_item_budget=1_024,
        fragment_byte_budget=1_048_576,
        cursor_byte_budget=8_192,
    )
    wide_sink = _FragmentSink(example_count)
    wide = planner_module.stream_online_pack_plan_fragments(
        lambda: iter(examples),
        fragment_sink=wide_sink,
        global_max_length=4,
        lookahead=8,
        max_packs_per_fragment=400,
        fragment_item_budget=2_048,
        fragment_byte_budget=2_097_152,
        cursor_byte_budget=8_192,
    )

    assert narrow_sink.seen == bytearray([1]) * example_count
    assert wide_sink.seen == bytearray([1]) * example_count
    assert narrow_sink.memberships == wide_sink.memberships
    assert narrow.terminal_cursor.source_prefix_sha256 == (
        wide.terminal_cursor.source_prefix_sha256
    )
    assert narrow.terminal_cursor.emitted_prefix_sha256 == (
        wide.terminal_cursor.emitted_prefix_sha256
    )
    assert narrow.fragment_count > wide.fragment_count
    assert narrow.max_fragment_items_observed <= 1_024
    assert narrow.max_fragment_bytes_observed <= 1_048_576
    assert wide.max_fragment_items_observed <= 2_048
    assert wide.max_fragment_bytes_observed <= 2_097_152
    assert narrow.serialized_size_bytes <= 2_048
    assert wide.serialized_size_bytes <= 2_048


def test_online_fragment_boundaries_do_not_change_irregular_membership() -> None:
    examples = _examples(tuple((ordinal * 7) % 9 + 1 for ordinal in range(2_000)))
    narrow_sink = _FragmentSink(len(examples))
    narrow = planner_module.stream_online_pack_plan_fragments(
        lambda: iter(examples),
        fragment_sink=narrow_sink,
        global_max_length=10,
        lookahead=8,
        max_packs_per_fragment=1_000,
        fragment_item_budget=96,
        fragment_byte_budget=65_536,
    )
    wide_sink = _FragmentSink(len(examples))
    wide = planner_module.stream_online_pack_plan_fragments(
        lambda: iter(examples),
        fragment_sink=wide_sink,
        global_max_length=10,
        lookahead=8,
        max_packs_per_fragment=1_000,
        fragment_item_budget=192,
        fragment_byte_budget=131_072,
    )

    assert narrow_sink.memberships == wide_sink.memberships
    assert narrow.terminal_cursor.emitted_prefix_sha256 == (
        wide.terminal_cursor.emitted_prefix_sha256
    )


def test_online_fragment_stream_invokes_source_once_and_yields_each_input_once() -> (
    None
):
    lengths = tuple(
        0 if ordinal % 37 == 0 else 11 if ordinal % 53 == 0 else (ordinal * 5) % 9 + 1
        for ordinal in range(500)
    )
    examples = _examples(lengths)
    factory_calls = 0
    source_yields = 0

    def source_factory():
        nonlocal factory_calls, source_yields
        factory_calls += 1
        for example in examples:
            source_yields += 1
            yield example

    fragments: list[PackPlan] = []
    receipt = planner_module.stream_online_pack_plan_fragments(
        source_factory,
        fragment_sink=fragments.append,
        global_max_length=10,
        lookahead=8,
        max_packs_per_fragment=7,
        fragment_item_budget=64,
        fragment_byte_budget=65_536,
    )

    assert factory_calls == 1
    assert source_yields == len(examples)
    assert receipt.source_input_count == len(examples)
    planner_module.verify_pack_plan_stream_fragments(receipt, fragments)


def test_stream_receipt_strict_roundtrip_and_ordered_chain_verification() -> None:
    fragments: list[PackPlan] = []
    receipt = planner_module.stream_online_pack_plan_fragments(
        lambda: iter(_examples((6, 2, 5, 4, 3, 8, 1, 7, 2))),
        fragment_sink=fragments.append,
        global_max_length=10,
        lookahead=3,
        max_packs_per_fragment=1,
        seed=2026,
    )

    decoded = PackPlanStreamReceipt.from_json(receipt.to_json())
    assert decoded == receipt
    assert decoded.to_json() == receipt.to_json()
    planner_module.verify_pack_plan_stream_fragments(decoded, fragments)

    other_fragments: list[PackPlan] = []
    other_receipt = planner_module.stream_online_pack_plan_fragments(
        lambda: iter(_examples((3, 3, 3))),
        fragment_sink=other_fragments.append,
        global_max_length=6,
        lookahead=2,
        max_packs_per_fragment=1,
    )
    invalid_fragment_sequences = (
        fragments[1:],
        [fragments[1], fragments[0], *fragments[2:]],
        [fragments[0], fragments[0], *fragments[1:]],
        [*fragments[:-1], other_fragments[0]],
    )
    for invalid in invalid_fragment_sequences:
        with pytest.raises(PackingContractError):
            planner_module.verify_pack_plan_stream_fragments(decoded, invalid)

    with pytest.raises(PackingContractError):
        planner_module.verify_pack_plan_stream_fragments(other_receipt, fragments)

    unknown = decoded.to_dict()
    unknown["unexpected"] = True
    with pytest.raises(PackingContractError) as exc_info:
        PackPlanStreamReceipt.from_dict(unknown)
    assert exc_info.value.code == "packing.pack_plan_fields"

    corrupt = decoded.to_dict()
    corrupt["authentication"]["sha256"] = "0" * 64
    with pytest.raises(PackingContractError) as exc_info:
        PackPlanStreamReceipt.from_dict(corrupt)
    assert exc_info.value.code == "packing.pack_plan_authentication"


def test_online_cursor_rejects_identity_that_exceeds_serialized_byte_budget() -> None:
    with pytest.raises(PackingContractError) as exc_info:
        create_pack_plan(
            _examples((1,)),
            global_max_length=1,
            policy=ONLINE_WINDOW_BINPACK,
            lookahead=1,
            cursor_byte_budget=2_048,
            max_packs=1,
            intra_image_order_identity_getter=lambda _example: "x" * 4_096,
        )

    assert exc_info.value.code == "packing.pack_plan_cursor_byte_bound"


def test_replay_rejects_intra_image_order_identity_drift() -> None:
    examples = _examples((5, 5, 4))
    plan = create_pack_plan(
        examples,
        global_max_length=10,
        policy=WINDOW_BINPACK,
        window_size=3,
    )
    changed = list(examples)
    changed[1] = replace(changed[1], row_ids=("row-z", "row-y"))

    with pytest.raises(PackingContractError) as exc_info:
        replay_pack_plan(plan, changed)

    assert exc_info.value.code == "packing.pack_plan_replay_identity"


def test_real_encoded_example_replay_rejects_complete_supervision_semantic_drift() -> (
    None
):
    example = _real_encoded_example()
    plan = create_pack_plan(
        (example,),
        global_max_length=16,
        policy=SOURCE_ORDER_NEXT_FIT,
    )
    changed_target = replace(
        example.supervised_token_spans[0],
        coordinate_target=CoordinateLossTarget(bbox=(10, 20, 80, 90), slot_index=0),
    )
    changed_ignored = replace(example.ignored_token_spans[0], source="changed-source")

    for changed in (
        replace(example, supervised_token_spans=(changed_target,)),
        replace(example, ignored_token_spans=(changed_ignored,)),
    ):
        with pytest.raises(PackingContractError) as exc_info:
            replay_pack_plan(plan, (changed,))
        assert exc_info.value.code == "packing.pack_plan_replay_identity"


def test_real_encoded_example_replay_authenticates_every_execution_field() -> None:
    example = _real_encoded_example()
    plan = create_pack_plan(
        (example,),
        global_max_length=16,
        policy=SOURCE_ORDER_NEXT_FIT,
    )
    changed_pixel_values = example.image_encoding.pixel_values.clone()
    changed_pixel_values[0, 0] += 1
    changed_grid = example.image_encoding.image_grid_thw_tensor.clone()
    changed_grid[0, 2] += 1
    changed_supervision = replace(
        example.supervised_token_spans[0],
        field="x2",
    )
    changed_ignored = replace(example.ignored_token_spans[0], source="padding")
    mutations = (
        replace(example, example_id="changed-example"),
        replace(example, chat_text=f"{example.chat_text}changed"),
        replace(example, base_input_ids=(999, *example.base_input_ids[1:])),
        replace(example, input_ids=(999, *example.input_ids[1:])),
        replace(
            example, base_offset_mapping=((0, 2), *example.base_offset_mapping[1:])
        ),
        replace(
            example, base_to_physical_start=(1, *example.base_to_physical_start[1:])
        ),
        replace(example, image_token_count=example.image_token_count + 1),
        replace(
            example,
            image_encoding=replace(
                example.image_encoding,
                pixel_values=changed_pixel_values,
            ),
        ),
        replace(
            example,
            image_encoding=replace(
                example.image_encoding,
                image_grid_thw_tensor=changed_grid,
            ),
        ),
        replace(
            example,
            image_encoding=replace(
                example.image_encoding,
                plan=replace(example.image_encoding.plan, raw_pixels=783),
            ),
        ),
        replace(
            example,
            image_encoding=replace(
                example.image_encoding,
                execution_evidence=QwenExecutedImageEvidence(
                    media_sha256="2" * 64,
                    decoded_width=28,
                    decoded_height=28,
                    observed_image_grid_thw=(1, 2, 2),
                ),
            ),
        ),
        replace(
            example,
            image_encoding=replace(
                example.image_encoding,
                image_processor=object(),
            ),
        ),
        replace(
            example,
            assistant_content_start_char=example.assistant_content_start_char + 1,
        ),
        replace(example, image_pad_base_index=example.image_pad_base_index + 1),
        replace(
            example,
            image_pad_physical_start=example.image_pad_physical_start + 1,
        ),
        replace(
            example,
            image_pad_physical_end=example.image_pad_physical_end + 1,
        ),
        replace(example, supervised_token_spans=(changed_supervision,)),
        replace(example, ignored_token_spans=(changed_ignored,)),
        replace(example, global_max_length=example.global_max_length + 1),
    )

    for changed in mutations:
        with pytest.raises(PackingContractError) as exc_info:
            replay_pack_plan(plan, (changed,))
        assert exc_info.value.code == "packing.pack_plan_replay_identity"


def test_encoded_tensor_identity_fails_before_exceeding_digest_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = _real_encoded_example()
    monkeypatch.setattr(planner_module, "DEFAULT_PACK_PLAN_CONTENT_BYTE_BUDGET", 16)

    with pytest.raises(PackingContractError) as exc_info:
        create_pack_plan(
            (example,),
            global_max_length=16,
            policy=SOURCE_ORDER_NEXT_FIT,
        )

    assert exc_info.value.code == "packing.pack_plan_encoded_semantics_bound"


def test_source_order_replay_matches_complete_rematerialized_pack_and_identities() -> (
    None
):
    original = _real_encoded_example()
    peer = replace(
        original,
        example_id="real-ex-peer",
        image_encoding=replace(
            original.image_encoding,
            plan=replace(original.image_encoding.plan, example_id="real-ex-peer"),
        ),
    )
    examples = (original, peer)
    plan = create_pack_plan(
        examples,
        global_max_length=8,
        policy=SOURCE_ORDER_NEXT_FIT,
    )
    rematerialized = tuple(
        replace(
            example,
            image_encoding=replace(
                example.image_encoding,
                pixel_values=example.image_encoding.pixel_values.clone(),
                image_grid_thw_tensor=(
                    example.image_encoding.image_grid_thw_tensor.clone()
                ),
            ),
            supervised_token_spans=tuple(
                replace(span) for span in example.supervised_token_spans
            ),
            ignored_token_spans=tuple(
                replace(span) for span in example.ignored_token_spans
            ),
        )
        for example in examples
    )

    replayed = replay_pack_plan(plan, rematerialized)
    expected_packs = plan_packed_sequences(rematerialized, global_max_length=8)
    rematerialized_plan = create_pack_plan(
        rematerialized,
        global_max_length=8,
        policy=SOURCE_ORDER_NEXT_FIT,
    )

    assert replayed == expected_packs
    assert [pack.to_artifact_dict() for pack in replayed] == [
        pack.to_artifact_dict() for pack in expected_packs
    ]
    assert plan.inputs == rematerialized_plan.inputs


def test_real_encoded_example_roundtrip_replays_bit_exact_supervision_identity() -> (
    None
):
    example = _real_encoded_example()

    plan = PackPlan.from_json(
        create_pack_plan(
            (example,),
            global_max_length=16,
            policy=SOURCE_ORDER_NEXT_FIT,
        ).to_json()
    )
    replayed = replay_pack_plan(plan, (example,))

    assert replayed[0].input_ids == example.input_ids
    assert replayed[0].segments[0].example_id == example.example_id
    assert plan.inputs[0].encoded_example_semantics_sha256 == (
        "15bef7472608ee0a6f257578ccbc2d086312fd68aa651065465c6d52c51332c7"
    )


def test_utilization_metrics_are_exact_and_window_is_bounded() -> None:
    examples = _examples((6, 6, 4, 4))
    source = create_pack_plan(
        examples,
        global_max_length=10,
        policy=SOURCE_ORDER_NEXT_FIT,
    )
    windowed = create_pack_plan(
        examples,
        global_max_length=10,
        policy=WINDOW_BINPACK,
        window_size=4,
    )

    assert (source.used_tokens, source.capacity_tokens, source.tail_waste) == (
        20,
        30,
        10,
    )
    assert source.utilization == pytest.approx(2 / 3)
    assert (windowed.used_tokens, windowed.capacity_tokens, windowed.tail_waste) == (
        20,
        20,
        0,
    )
    assert windowed.utilization == 1.0
    assert windowed.max_pending_items_observed == 4


def _examples(lengths: tuple[int, ...]) -> tuple[FakeEncodedExample, ...]:
    next_token = 10
    examples: list[FakeEncodedExample] = []
    for ordinal, length in enumerate(lengths):
        input_ids = tuple(range(next_token, next_token + length))
        next_token += max(length, 1) + 10
        examples.append(
            FakeEncodedExample(
                example_id=f"ex-{ordinal}",
                input_ids=input_ids,
                row_ids=(f"row-{ordinal}-a", f"row-{ordinal}-b"),
            )
        )
    return tuple(examples)


@dataclass(frozen=True)
class FakeEncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    row_ids: tuple[str, ...]

    @property
    def intra_image_order_identity(self) -> str:
        material = json.dumps(self.row_ids, separators=(",", ":")).encode()
        return f"test-row-order-sha256:{hashlib.sha256(material).hexdigest()}"


class _FragmentSink:
    def __init__(self, example_count: int) -> None:
        self.seen = bytearray(example_count)
        self.memberships: list[tuple[int, ...]] = []
        self._predecessor_sha256: str | None = None

    def __call__(self, fragment: PackPlan) -> None:
        assert fragment.predecessor_plan_sha256 == self._predecessor_sha256
        for pack in fragment.packs:
            self.memberships.append(pack.input_ordinals)
            for ordinal in pack.input_ordinals:
                assert self.seen[ordinal] == 0
                self.seen[ordinal] = 1
        for rejected in fragment.rejected_examples:
            assert self.seen[rejected.input_ordinal] == 0
            self.seen[rejected.input_ordinal] = 1
        self._predecessor_sha256 = fragment.canonical_sha256


def _real_encoded_example() -> EncodedExample:
    image_plan = QwenNoResizeImagePlan(
        example_id="real-ex",
        image_path=Path("fixtures/real-ex.png"),
        width=28,
        height=28,
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        required_spatial_factor=28,
        raw_pixels=784,
        raw_patch_rows=4,
        expected_pixel_values_width=1176,
        image_grid_thw=(1, 2, 2),
        merged_visual_tokens=1,
        max_raw_pixels=784,
        max_merged_visual_tokens=1,
        image_content_sha256="1" * 64,
        decoded_width=28,
        decoded_height=28,
    )
    coordinate_span = EncodedTokenSpan(
        token_type="coordinate",
        text="<|coord_10|>",
        char_start=0,
        char_end=12,
        chat_char_start=16,
        chat_char_end=28,
        base_token_start=2,
        base_token_end=3,
        physical_token_start=2,
        physical_token_end=3,
        token_ids=(102,),
        object_id="object-0",
        field="x1",
        source="bbox",
        coordinate_target=CoordinateLossTarget(
            bbox=(10, 20, 70, 90),
            slot_index=0,
        ),
    )
    ignored_span = EncodedTokenSpan(
        token_type="ignored",
        text=" ",
        char_start=12,
        char_end=13,
        chat_char_start=28,
        chat_char_end=29,
        base_token_start=3,
        base_token_end=4,
        physical_token_start=3,
        physical_token_end=4,
        token_ids=(103,),
        object_id=None,
        field=None,
        source="separator",
    )
    return EncodedExample(
        example_id="real-ex",
        chat_text="<|im_start|>assistant\n<|coord_10|> ",
        base_input_ids=(100, 101, 102, 103),
        input_ids=(100, 101, 102, 103),
        base_offset_mapping=((0, 1), (1, 2), (16, 28), (28, 29)),
        base_to_physical_start=(0, 1, 2, 3),
        image_token_count=1,
        image_encoding=QwenImageEncoding(
            plan=image_plan,
            pixel_values=torch.arange(4 * 1176, dtype=torch.float32).reshape(4, 1176),
            image_grid_thw_tensor=torch.tensor([[1, 2, 2]], dtype=torch.int64),
        ),
        assistant_content_start_char=16,
        image_pad_base_index=1,
        image_pad_physical_start=1,
        image_pad_physical_end=2,
        supervised_token_spans=(coordinate_span,),
        ignored_token_spans=(ignored_span,),
        global_max_length=16,
    )


def _reauthenticate(payload: dict[str, object]) -> dict[str, object]:
    canonical_payload = {
        key: value for key, value in payload.items() if key != "authentication"
    }
    payload["authentication"]["sha256"] = hashlib.sha256(
        json.dumps(
            canonical_payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return payload
