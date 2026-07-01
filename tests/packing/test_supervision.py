from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.common.errors import PackingContractError
from src.config.loader import load_train_config
from src.data import load_raw_examples
from src.packing.planner import plan_packed_sequences
from src.packing.supervision import build_packed_supervision
from src.qwen.encoding import encode_rendered_example
from src.qwen.loading import load_qwen_components
from src.templates import render_example


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


def test_packed_supervision_remaps_encoded_spans_to_pack_positions() -> None:
    examples = (
        FakeEncodedExample(
            "ex-0",
            (10, 11, 12, 13),
            (
                FakeTokenSpan("schema", base=1, physical=1, token_id=11, text="A"),
                FakeTokenSpan("eos", base=3, physical=3, token_id=13, text="B"),
            ),
        ),
        FakeEncodedExample(
            "ex-1",
            (20, 21, 22),
            (
                FakeTokenSpan("coordinate", base=1, physical=1, token_id=21, text="C"),
            ),
        ),
    )
    packs = plan_packed_sequences(examples, global_max_length=7)

    supervision = build_packed_supervision(packs, examples)

    assert [atom.example_id for atom in supervision.atoms] == ["ex-0", "ex-0", "ex-1"]
    assert [atom.target_position for atom in supervision.atoms] == [1, 3, 5]
    assert [atom.logits_position for atom in supervision.atoms] == [0, 2, 4]
    assert [atom.segment_index for atom in supervision.atoms] == [0, 0, 1]
    assert [atom.logical_target_position for atom in supervision.atoms] == [1, 3, 1]
    assert supervision.atoms[-1].pack_index == 0
    assert supervision.atoms[-1].token_type == "coordinate"
    assert supervision.atoms[-1].token_id == 21
    assert supervision.to_artifact_dict()["atom_count"] == 3


def test_packed_supervision_maps_real_encoded_smoke_examples() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    components = load_qwen_components(resolved.config, load_model=False)
    examples = []
    for raw in load_raw_examples(resolved.config.data.train):
        rendered = render_example(raw, resolved.config.template)
        examples.append(
            encode_rendered_example(
                raw,
                rendered,
                components=components,
                processor_config=resolved.config.model.processor,
                global_max_length=resolved.config.packing.global_max_length,
            )
        )
    packs = plan_packed_sequences(tuple(examples), global_max_length=12_000)

    supervision = build_packed_supervision(packs, tuple(examples))

    assert len(packs) == 1
    assert [segment.example_id for segment in packs[0].segments] == [
        "coco2017_train_000000000030__smoke2obj",
        "coco2017_train_000000000036__smoke2obj",
    ]
    assert len(supervision.omitted_atoms) == 0
    assert supervision.atoms[0].target_position == examples[0].supervised_token_spans[0].physical_token_start
    second_segment = packs[0].segments[1]
    first_second_atom = next(atom for atom in supervision.atoms if atom.example_id == examples[1].example_id)
    assert first_second_atom.target_position == (
        second_segment.start + examples[1].supervised_token_spans[0].physical_token_start
    )
    assert first_second_atom.logits_position >= second_segment.start


def test_boundary_target_is_omitted_not_shifted_to_previous_segment() -> None:
    examples = (
        FakeEncodedExample("ex-0", (10, 11), (FakeTokenSpan("schema", base=1, physical=1),)),
        FakeEncodedExample(
            "ex-1",
            (20, 21),
            (
                FakeTokenSpan(
                    "schema",
                    base=0,
                    physical=0,
                    token_id=20,
                    text="boundary",
                    field="first",
                    source="test-source",
                ),
            ),
        ),
    )
    packs = plan_packed_sequences(examples, global_max_length=4)

    supervision = build_packed_supervision(packs, examples)

    assert [atom.example_id for atom in supervision.atoms] == ["ex-0"]
    assert len(supervision.omitted_atoms) == 1
    omitted = supervision.omitted_atoms[0]
    assert omitted.example_id == "ex-1"
    assert omitted.target_position == 2
    assert omitted.logits_position == 1
    assert omitted.token_id == 20
    assert omitted.text == "boundary"
    assert omitted.logical_target_end == 1
    assert omitted.target_end == 3
    assert omitted.field == "first"
    assert omitted.source == "test-source"
    assert omitted.reason == "logits_position_crosses_segment_boundary"


def test_boundary_drop_omits_only_the_crossing_atom_in_multi_token_span() -> None:
    examples = (
        FakeEncodedExample("ex-0", (10, 11), ()),
        FakeEncodedExample(
            "ex-1",
            (20, 21, 22),
            (FakeMultiTokenSpan("desc_text", physical=0, token_ids=(20, 21)),),
        ),
    )
    packs = plan_packed_sequences(examples, global_max_length=5)

    supervision = build_packed_supervision(packs, examples)

    assert len(supervision.omitted_atoms) == 1
    assert supervision.omitted_atoms[0].target_position == 2
    assert [atom.target_position for atom in supervision.atoms] == [3]
    assert [atom.logits_position for atom in supervision.atoms] == [2]
    assert supervision.atoms[0].logical_target_position == 1


def test_missing_encoded_example_for_segment_fails() -> None:
    examples = (
        FakeEncodedExample("ex-0", (10, 11), (FakeTokenSpan("schema", base=1, physical=1),)),
    )
    packs = plan_packed_sequences(examples, global_max_length=4)

    with pytest.raises(PackingContractError) as exc_info:
        build_packed_supervision(packs, ())

    assert exc_info.value.code == "packing.segment_example_missing"


@pytest.mark.parametrize("physical_start", [2, 3])
def test_out_of_segment_logical_target_range_fails(physical_start: int) -> None:
    examples = (
        FakeEncodedExample(
            "ex-0",
            (10, 11),
            (FakeTokenSpan("schema", base=physical_start, physical=physical_start),),
        ),
        FakeEncodedExample("ex-1", (20, 21), ()),
    )
    packs = plan_packed_sequences(examples, global_max_length=4)

    with pytest.raises(PackingContractError) as exc_info:
        build_packed_supervision(packs, examples)

    assert exc_info.value.code == "packing.supervision_span_bounds"
    assert exc_info.value.context["logical_start"] == physical_start


@dataclass(frozen=True)
class FakeEncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    supervised_token_spans: tuple["FakeTokenSpan", ...]

    @property
    def input_length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class FakeTokenSpan:
    token_type: str
    base: int
    physical: int
    token_id: int = 999
    text: str = "x"
    object_id: str | None = None
    field: str | None = None
    source: str | None = None

    @property
    def physical_token_start(self) -> int:
        return self.physical

    @property
    def physical_token_end(self) -> int:
        return self.physical + 1

    @property
    def base_token_start(self) -> int:
        return self.base

    @property
    def base_token_end(self) -> int:
        return self.base + 1

    @property
    def token_ids(self) -> tuple[int, ...]:
        return (self.token_id,)


@dataclass(frozen=True)
class FakeMultiTokenSpan:
    token_type: str
    physical: int
    token_ids: tuple[int, ...]
    text: str = "xx"
    object_id: str | None = None
    field: str | None = None
    source: str | None = None

    @property
    def physical_token_start(self) -> int:
        return self.physical

    @property
    def physical_token_end(self) -> int:
        return self.physical + len(self.token_ids)

    @property
    def base_token_start(self) -> int:
        return self.physical

    @property
    def base_token_end(self) -> int:
        return self.physical + len(self.token_ids)
