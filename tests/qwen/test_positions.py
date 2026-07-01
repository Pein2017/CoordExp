from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from src.common.errors import QwenForwardContractError
from src.config.loader import load_train_config
from src.data import load_raw_examples
from src.packing.planner import plan_packed_sequences
from src.qwen.encoding import encode_rendered_example
from src.qwen.loading import load_qwen_components
from src.qwen.positions import (
    build_qwen_position_inputs,
    validate_qwen_4row_position_boundary,
)
from src.templates import render_example


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


def test_single_segment_positions_follow_qwen_mrope_formula() -> None:
    example = FakeEncodedExample(
        example_id="ex-0",
        input_ids=(10, 11, 12, 151655, 151655, 151655, 151655, 151655, 151655, 13, 14),
        image_pad_physical_start=3,
        image_pad_physical_end=9,
        image_grid_thw=(1, 4, 6),
        merge_size=2,
    )
    pack = plan_packed_sequences((example,), global_max_length=20)[0]

    positions = build_qwen_position_inputs(pack, (example,))

    assert tuple(positions.position_ids.shape) == (4, 1, 11)
    assert positions.position_ids.dtype == torch.long
    assert positions.segment_boundaries == (0, 11)
    assert positions.reset_points == (0,)
    assert positions.max_segment_length == 11
    assert positions.position_ids[0, 0].tolist() == list(range(11))
    assert positions.position_ids[1, 0].tolist() == [0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7]
    assert positions.position_ids[2, 0].tolist() == [0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7]
    assert positions.position_ids[3, 0].tolist() == [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7]
    artifact = positions.to_artifact_dict()
    assert artifact["position_ids_shape"] == [4, 1, 11]
    assert artifact["segments"][0]["text_start"] == 0
    assert artifact["segments"][0]["mrope_row_max"] == [7, 7, 7]


def test_packed_positions_reset_at_segment_boundaries() -> None:
    examples = (
        FakeEncodedExample(
            example_id="ex-0",
            input_ids=(10, 11, 12, 151655, 151655, 151655, 151655, 151655, 151655, 13, 14),
            image_pad_physical_start=3,
            image_pad_physical_end=9,
            image_grid_thw=(1, 4, 6),
            merge_size=2,
        ),
        FakeEncodedExample(
            example_id="ex-1",
            input_ids=(20, 21, 151655, 151655, 151655, 151655, 151655, 151655, 22),
            image_pad_physical_start=2,
            image_pad_physical_end=8,
            image_grid_thw=(1, 4, 6),
            merge_size=2,
        ),
    )
    pack = plan_packed_sequences(examples, global_max_length=20)[0]

    positions = build_qwen_position_inputs(pack, examples, expected_cu_seq_lens=(0, 11, 20))

    second_segment = pack.segments[1]
    assert positions.segment_boundaries == (0, 11, 20)
    assert positions.reset_points == (0, 11)
    assert positions.position_ids[0, 0, 0].item() == 0
    assert positions.position_ids[0, 0, second_segment.start].item() == 0
    assert positions.position_ids[0, 0, second_segment.start:second_segment.end].tolist() == list(
        range(second_segment.length)
    )
    assert positions.position_ids[1, 0, second_segment.start:second_segment.start + 4].tolist() == [
        0,
        1,
        2,
        2,
    ]


def test_cu_seq_lens_must_match_segment_boundaries() -> None:
    example = FakeEncodedExample(
        example_id="ex-0",
        input_ids=(10, 151655, 151655, 151655, 151655),
        image_pad_physical_start=1,
        image_pad_physical_end=5,
        image_grid_thw=(1, 4, 4),
        merge_size=2,
    )
    pack = plan_packed_sequences((example,), global_max_length=10)[0]

    with pytest.raises(QwenForwardContractError) as exc_info:
        build_qwen_position_inputs(pack, (example,), expected_cu_seq_lens=(0, 4))

    assert exc_info.value.code == "qwen.position_boundary_mismatch"
    assert exc_info.value.context["segment_boundaries"] == [0, 5]
    assert exc_info.value.context["expected_cu_seq_lens"] == [0, 4]


def test_qwen_4row_boundary_validation_accepts_installed_forward() -> None:
    validation = validate_qwen_4row_position_boundary()

    assert validation.row_meaning == ("text", "temporal", "height", "width")
    assert validation.checks["has_4row_branch"] is True
    assert validation.checks["splits_text_row"] is True
    assert validation.checks["uses_remaining_rows_for_rotary"] is True
    assert validation.checks["routes_text_row_to_attention"] is True
    assert validation.checks["routes_mrope_rows_to_rotary"] is True


def test_qwen_4row_boundary_validation_rejects_changed_forward() -> None:
    broken_forward = """
def forward(self, input_ids=None, position_ids=None):
    if position_ids is not None and position_ids.ndim == 3:
        text_position_ids = position_ids[0]
    attention_mask = create_causal_mask(position_ids=text_position_ids)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
"""

    with pytest.raises(QwenForwardContractError) as exc_info:
        validate_qwen_4row_position_boundary(forward_source=broken_forward)

    assert exc_info.value.code == "qwen.position_boundary_semantics"
    assert exc_info.value.context["missing_checks"] == [
        "has_4row_branch",
        "uses_remaining_rows_for_rotary",
    ]


def test_malformed_position_inputs_raise_qwen_contract_errors() -> None:
    bad_grid = FakeEncodedExample(
        example_id="bad-grid",
        input_ids=(10, 151655, 151655, 151655, 151655),
        image_pad_physical_start=1,
        image_pad_physical_end=5,
        image_grid_thw=("bad", 4, 4),
        merge_size=2,
    )
    bad_merge = FakeEncodedExample(
        example_id="bad-merge",
        input_ids=(10, 151655, 151655, 151655, 151655),
        image_pad_physical_start=1,
        image_pad_physical_end=5,
        image_grid_thw=(1, 4, 4),
        merge_size="bad",
    )

    with pytest.raises(QwenForwardContractError) as exc_info:
        build_qwen_position_inputs(
            plan_packed_sequences((bad_grid,), global_max_length=10)[0],
            (bad_grid,),
        )
    assert exc_info.value.code == "qwen.position_image_grid"

    with pytest.raises(QwenForwardContractError) as exc_info:
        build_qwen_position_inputs(
            plan_packed_sequences((bad_merge,), global_max_length=10)[0],
            (bad_merge,),
        )
    assert exc_info.value.code == "qwen.position_merge_size"

    good = FakeEncodedExample(
        example_id="bad-cu-seq",
        input_ids=(10, 151655, 151655, 151655, 151655),
        image_pad_physical_start=1,
        image_pad_physical_end=5,
        image_grid_thw=(1, 4, 4),
        merge_size=2,
    )
    with pytest.raises(QwenForwardContractError) as exc_info:
        build_qwen_position_inputs(
            plan_packed_sequences((good,), global_max_length=10)[0],
            (good,),
            expected_cu_seq_lens=(0, "bad"),
        )
    assert exc_info.value.code == "qwen.position_cu_seq_lens"

    with pytest.raises(QwenForwardContractError) as exc_info:
        build_qwen_position_inputs(
            plan_packed_sequences((good,), global_max_length=10)[0],
            (good,),
            expected_cu_seq_lens=0,
        )
    assert exc_info.value.code == "qwen.position_cu_seq_lens"


def test_image_span_must_match_actual_contiguous_image_token_run() -> None:
    example = FakeEncodedExample(
        example_id="shifted-image-span",
        input_ids=(10, 11, 12, 151655, 151655, 151655, 151655, 151655, 151655, 13),
        image_pad_physical_start=2,
        image_pad_physical_end=8,
        image_grid_thw=(1, 4, 6),
        merge_size=2,
    )
    pack = plan_packed_sequences((example,), global_max_length=20)[0]

    with pytest.raises(QwenForwardContractError) as exc_info:
        build_qwen_position_inputs(pack, (example,))

    assert exc_info.value.code == "qwen.position_image_token_span"


def test_real_smoke_packed_positions_match_upstream_per_segment_helper() -> None:
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
    examples = tuple(examples)
    pack = plan_packed_sequences(examples, global_max_length=12_000)[0]
    boundaries = tuple([0, *[segment.end for segment in pack.segments]])

    positions = build_qwen_position_inputs(pack, examples, expected_cu_seq_lens=boundaries)

    assert tuple(positions.position_ids.shape) == (4, 1, pack.length)
    assert positions.segment_boundaries == boundaries
    assert [positions.position_ids[0, 0, segment.start].item() for segment in pack.segments] == [
        0,
        0,
    ]
    for segment in pack.segments:
        example = next(item for item in examples if item.example_id == segment.example_id)
        upstream = _upstream_get_rope_index(example, components)
        observed = positions.position_ids[1:, 0, segment.start:segment.end]
        assert torch.equal(observed, upstream[:, 0])

    whole_pack_upstream = _upstream_get_rope_index_for_pack(pack, examples, components)
    second_start = pack.segments[1].start
    assert whole_pack_upstream[0, 0, second_start].item() != 0


def _upstream_get_rope_index(example: object, components: object) -> torch.Tensor:
    return _call_upstream_get_rope_index(
        input_ids=tuple(int(item) for item in getattr(example, "input_ids")),
        image_grids=(tuple(int(item) for item in example.image_encoding.image_grid_thw),),
        components=components,
    )


def _upstream_get_rope_index_for_pack(
    pack: object,
    examples: tuple[object, ...],
    components: object,
) -> torch.Tensor:
    examples_by_id = {example.example_id: example for example in examples}
    image_grids = tuple(
        tuple(
            int(item)
            for item in examples_by_id[segment.example_id].image_encoding.image_grid_thw
        )
        for segment in pack.segments
    )
    return _call_upstream_get_rope_index(
        input_ids=tuple(int(item) for item in pack.input_ids),
        image_grids=image_grids,
        components=components,
    )


def _call_upstream_get_rope_index(
    *,
    input_ids: tuple[int, ...],
    image_grids: tuple[tuple[int, int, int], ...],
    components: object,
) -> torch.Tensor:
    fake_model = SimpleNamespace(config=components.config)
    position_ids, _ = Qwen3VLModel.get_rope_index(
        fake_model,
        input_ids=torch.tensor([list(input_ids)], dtype=torch.long),
        image_grid_thw=torch.tensor(image_grids, dtype=torch.long),
        video_grid_thw=None,
        attention_mask=None,
    )
    return position_ids


@dataclass(frozen=True)
class FakeEncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    image_pad_physical_start: int
    image_pad_physical_end: int
    image_grid_thw: tuple[object, object, object]
    merge_size: object

    @property
    def image_token_count(self) -> int:
        return self.image_pad_physical_end - self.image_pad_physical_start

    @property
    def image_encoding(self) -> "FakeImageEncoding":
        return FakeImageEncoding(
            image_grid_thw=self.image_grid_thw,
            merged_visual_tokens=self.image_token_count,
            plan=FakeImagePlan(merge_size=self.merge_size),
        )


@dataclass(frozen=True)
class FakeImageEncoding:
    image_grid_thw: tuple[object, object, object]
    merged_visual_tokens: int
    plan: "FakeImagePlan"


@dataclass(frozen=True)
class FakeImagePlan:
    merge_size: object
