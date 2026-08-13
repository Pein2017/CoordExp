from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from scripts.research import human13_live_payload as payload
from scripts.research import run_human13_k_union_overfit as runner


@dataclass(frozen=True)
class _ImageEncoding:
    image_grid_thw: tuple[int, int, int] = (1, 4, 4)
    merged_visual_tokens: int = 4
    plan: object = field(default_factory=lambda: SimpleNamespace(merge_size=2))


@dataclass(frozen=True)
class _Encoded:
    example_id: str
    input_ids: tuple[int, ...]
    human13_row_bindings: tuple[runner.Human13EncodedRowBinding, ...]
    image_pad_physical_start: int = 1
    image_pad_physical_end: int = 5
    image_token_id: int = 151655

    @property
    def image_encoding(self) -> _ImageEncoding:
        return _ImageEncoding()


def _segment(
    segment_id: str,
    role: runner.LogicalRole,
    *,
    length: int = 8,
    start: int = 6,
    family: runner.LossFamily = "h",
    unit_id: str = "owner-1",
    row_id: str = "row-1",
    mask: tuple[bool, ...] = (True, True),
) -> runner.LogicalPanelSegment:
    ids = (10, 151655, 151655, 151655, 151655, 20, 21, 22)[:length]
    binding = runner.Human13EncodedRowBinding(
        family=family,
        unit_id=unit_id,
        manifest_row_id=row_id,
        token_start=start,
        token_end=start + len(mask),
        target_token_mask=mask,
    )
    return runner.LogicalPanelSegment(
        segment_id=segment_id,
        image_id=1,
        role=role,
        encoded_example=_Encoded(segment_id, ids, (binding,)),
    )


def _materialized(segments: tuple[runner.LogicalPanelSegment, ...], *, a4=(), total=0):
    class Materialized:
        def __init__(self):
            self.segments = segments
            self.a4_segments = tuple(a4)
            self.a4_aggregate_lengths = ((1, total),) if a4 else ()
            self.calls = []

        def preflight(self, global_max_length=12_000, *, enforce_a4_aggregate=False):
            self.calls.append((global_max_length, enforce_a4_aggregate))
            if enforce_a4_aggregate and total > global_max_length:
                raise ValueError("A4 aggregate length exceeds 12,000")

    return Materialized()


def _sealed() -> object:
    manifest = SimpleNamespace(
        full_panel=True,
        arms=(SimpleNamespace(arm_id="A1"),),
        denominators=SimpleNamespace(
            target_owner_count=1,
            target_image_count=1,
            replay_owner_count=1,
            duplicate_image_count=1,
        ),
        binding=SimpleNamespace(panel=SimpleNamespace(owner_count=1)),
        images=(),
    )
    return runner.SealedHuman13Manifest(manifest=manifest, manifest_sha256="f" * 64)


def test_payload_derives_prompt_offset_causal_sites_and_empty_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = _materialized((_segment("a1", "a1_full_h"),))
    observed: dict[str, object] = {}

    def build_execution_plan(
        sealed,
        *,
        arm_id,
        packed_plan,
        sites_by_pack,
        a6_donor_binding=None,
        a8_census_binding=None,
    ):
        observed.update(
            arm_id=arm_id, packed_plan=packed_plan, sites_by_pack=sites_by_pack,
            a6_donor_binding=a6_donor_binding,
            a8_census_binding=a8_census_binding,
        )
        return "execution"

    monkeypatch.setattr(runner, "build_execution_plan", build_execution_plan)
    result = payload.build_live_payload(
        sealed_manifest=_sealed(),
        materialized_segments=materialized,
        arm_id="A1",
        expected_vocab_size=128,
        vocab_groups=SimpleNamespace(vocab_size=128),
    )

    site = result.sites_by_pack[0][0]
    assert site.logits_positions == (5, 6)
    assert site.target_token_ids == (21, 22)
    assert site.objective == "owner_ce"
    assert result.token_sequences[0].pack_index == 0
    assert result.token_sequences[0].atoms == ()
    assert observed["arm_id"] == "A1"


def test_payload_keeps_sites_separate_across_multiple_packs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segments = (
        _segment("a", "a1_full_h", length=8),
        _segment(
            "b",
            "source_replay",
            length=8,
            row_id="row-2",
            unit_id="owner-2",
            family="replay",
        ),
    )
    materialized = _materialized(segments)
    monkeypatch.setattr(runner, "build_execution_plan", lambda *args, **kwargs: "execution")

    result = payload.build_live_payload(
        sealed_manifest=_sealed(),
        materialized_segments=materialized,
        arm_id="A1",
        expected_vocab_size=128,
        vocab_groups=SimpleNamespace(vocab_size=128),
        global_max_length=8,
    )

    assert len(result.packed_plan.packs) == 2
    assert tuple(sorted(result.sites_by_pack)) == (0, 1)
    assert tuple(sequence.pack_index for sequence in result.token_sequences.values()) == (0, 1)


def test_a4_logical_group_may_span_multiple_physical_packs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        _segment("candidate-1", "a4_union", row_id="candidate-1", length=8),
        _segment("candidate-2", "a4_union", row_id="candidate-2", length=8),
    )
    materialized = _materialized(candidates, a4=candidates, total=16)
    monkeypatch.setattr(runner, "build_execution_plan", lambda *args, **kwargs: "execution")

    result = payload.build_live_payload(
        sealed_manifest=_sealed(),
        materialized_segments=materialized,
        arm_id="A4",
        expected_vocab_size=128,
        vocab_groups=SimpleNamespace(vocab_size=128),
        global_max_length=8,
    )

    assert len(result.packed_plan.packs) == 2
    assert {
        site.segment_id for sites in result.sites_by_pack.values() for site in sites
    } == {"candidate-1", "candidate-2"}
    assert materialized.calls == [(8, False)]


def test_a6_binding_is_passed_through_and_a8_margin_is_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a6 = _segment("a6", "a6_donor_h1")
    a6_binding = object()
    observed: dict[str, object] = {}

    def build_execution_plan(
        sealed,
        *,
        arm_id,
        packed_plan,
        sites_by_pack,
        a6_donor_binding=None,
        a8_census_binding=None,
    ):
        observed[arm_id] = a6_donor_binding
        return arm_id

    monkeypatch.setattr(runner, "build_execution_plan", build_execution_plan)
    result = payload.build_live_payload(
        sealed_manifest=_sealed(),
        materialized_segments=_materialized((a6,)),
        arm_id="A6",
        expected_vocab_size=128,
        vocab_groups=SimpleNamespace(vocab_size=128),
        a6_donor_binding=a6_binding,
    )
    assert observed["A6"] is a6_binding
    assert result.execution_plan == "A6"

    a8 = _segment("a8", "a8_full_h")
    a8_binding = SimpleNamespace(required_margin=0.25)
    monkeypatch.setattr(runner, "build_execution_plan", build_execution_plan)
    def build_a8_plan(
        sealed,
        *,
        arm_id,
        packed_plan,
        sites_by_pack,
        a6_donor_binding=None,
        a8_census_binding=None,
    ):
        assert a8_census_binding is a8_binding
        return "A8"

    monkeypatch.setattr(runner, "build_execution_plan", build_a8_plan)
    result = payload.build_live_payload(
        sealed_manifest=_sealed(),
        materialized_segments=_materialized((a8,)),
        arm_id="A8-prime",
        expected_vocab_size=128,
        vocab_groups=SimpleNamespace(vocab_size=128),
        a8_census_binding=a8_binding,
    )
    assert result.sites_by_pack[0][0].required_margin == pytest.approx(0.25)
