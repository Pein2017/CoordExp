from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import pytest
import torch
from PIL import Image

from src.common.errors import EncodingContractError
from src.config.loader import load_train_config
from src.config.models import ProcessorConfig
from src.data import ImageRef, RawExample, RawObject, SourceProvenance, load_raw_examples
from src.qwen.images import (
    attach_qwen_image_processor,
    build_no_resize_image_plan,
    encode_qwen_image,
    materialize_qwen_image_encoding,
    materialize_qwen_image_encoding_batch,
    plan_qwen_image,
)
from src.qwen.loading import QwenProcessorIdentity, load_qwen_components


FIXTURE_CONFIG = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack/config.yaml")


def test_real_smoke_image_encodes_with_processor_derived_no_resize_grid() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    example = load_raw_examples(resolved.config.data.train)[0]
    components = load_qwen_components(resolved.config, load_model=False)

    encoding = encode_qwen_image(
        example,
        components=components,
        processor_config=resolved.config.model.processor,
    )

    assert encoding.example_id == "coco2017_train_000000000030__smoke2obj"
    assert encoding.width == 1248
    assert encoding.height == 832
    assert encoding.required_spatial_factor == 32
    assert encoding.raw_pixels == 1_038_336
    assert encoding.image_grid_thw == (1, 52, 78)
    assert encoding.raw_patch_rows == 4056
    assert encoding.merged_visual_tokens == 1014
    assert tuple(encoding.pixel_values.shape) == (4056, 1536)
    assert tuple(encoding.image_grid_thw_tensor.shape) == (1, 3)
    assert encoding.to_artifact_dict()["image_grid_thw"] == [1, 52, 78]


def test_lazy_smoke_image_plan_defers_pixels_until_materialized() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    example = load_raw_examples(resolved.config.data.train)[0]
    components = load_qwen_components(resolved.config, load_model=False)

    planned = plan_qwen_image(
        example,
        components=components,
        processor_config=resolved.config.model.processor,
    )

    assert planned.image_grid_thw == (1, 52, 78)
    assert planned.pixel_values is None
    assert planned.image_grid_thw_tensor is None
    assert planned.to_artifact_dict()["pixel_values_materialized"] is False
    assert planned.to_artifact_dict()["pixel_values_shape"] is None

    materialized = materialize_qwen_image_encoding(planned)

    assert materialized.image_grid_thw == planned.image_grid_thw
    assert tuple(materialized.pixel_values.shape) == (4056, 1536)
    assert tuple(materialized.image_grid_thw_tensor.shape) == (1, 3)
    assert materialized.to_artifact_dict()["pixel_values_materialized"] is True


def test_real_smoke_image_batch_matches_single_image_materialization() -> None:
    resolved = load_train_config(FIXTURE_CONFIG)
    raw_examples = load_raw_examples(resolved.config.data.train)
    components = load_qwen_components(resolved.config, load_model=False)
    planned = tuple(
        plan_qwen_image(
            raw,
            components=components,
            processor_config=resolved.config.model.processor,
        )
        for raw in raw_examples
    )

    single_pixels = torch.cat(
        [
            materialize_qwen_image_encoding(encoding).pixel_values
            for encoding in planned
        ],
        dim=0,
    )
    single_grids = torch.cat(
        [
            materialize_qwen_image_encoding(encoding).image_grid_thw_tensor
            for encoding in planned
        ],
        dim=0,
    )
    batch_pixels, batch_grids = materialize_qwen_image_encoding_batch(planned)

    assert torch.equal(batch_pixels, single_pixels)
    assert torch.equal(batch_grids, single_grids)


def test_lazy_image_batch_materializes_once_and_preserves_order(tmp_path: Path) -> None:
    examples = (
        _raw_example(tmp_path / "a", width=96, height=64),
        _raw_example(tmp_path / "b", width=96, height=64),
    )
    processor = FakeProcessor(
        image_grid_thw=torch.tensor([[1, 4, 6], [1, 4, 6]], dtype=torch.long),
        pixel_values=torch.cat(
            [
                torch.full((24, 1536), 1.0, dtype=torch.float32),
                torch.full((24, 1536), 2.0, dtype=torch.float32),
            ],
            dim=0,
        ),
    )
    components = FakeComponents(
        processor_identity=_processor_identity(),
        processor=processor,
    )
    planned = tuple(
        plan_qwen_image(
            example,
            components=components,
            processor_config=_processor_config(),
        )
        for example in examples
    )

    pixel_values, image_grid_thw = materialize_qwen_image_encoding_batch(planned)

    assert processor.image_processor.batch_sizes == [2]
    assert tuple(pixel_values.shape) == (48, 1536)
    assert image_grid_thw.tolist() == [[1, 4, 6], [1, 4, 6]]
    assert torch.equal(pixel_values[:24], torch.full((24, 1536), 1.0))
    assert torch.equal(pixel_values[24:], torch.full((24, 1536), 2.0))


def test_lazy_image_plan_pickle_drops_and_reattaches_processor(tmp_path: Path) -> None:
    example = _raw_example(tmp_path, width=96, height=64)
    components = FakeComponents(
        processor_identity=_processor_identity(),
        processor=FakeProcessor(),
    )
    planned = plan_qwen_image(
        example,
        components=components,
        processor_config=_processor_config(),
    )

    cached = pickle.loads(pickle.dumps(planned))

    assert cached.pixel_values is None
    assert cached.image_grid_thw_tensor is None
    assert cached.image_processor is None
    with pytest.raises(EncodingContractError) as exc_info:
        materialize_qwen_image_encoding(cached)
    assert exc_info.value.code == "qwen.image_lazy_processor_missing"

    reattached = attach_qwen_image_processor(
        cached,
        components.processor.image_processor,
    )
    materialized = materialize_qwen_image_encoding(reattached)

    assert tuple(materialized.pixel_values.shape) == (24, 1536)
    assert tuple(materialized.image_grid_thw_tensor.shape) == (1, 3)


def test_invalid_no_resize_dimensions_fail_before_processor_call(tmp_path: Path) -> None:
    example = _raw_example(tmp_path, width=96, height=65)
    identity = _processor_identity()

    with pytest.raises(EncodingContractError, match="divisible"):
        build_no_resize_image_plan(
            example,
            processor_identity=identity,
            processor_config=_processor_config(),
        )


def test_raw_pixel_budget_fails_before_processor_call(tmp_path: Path) -> None:
    example = _raw_example(tmp_path, width=96, height=64)
    identity = _processor_identity()

    with pytest.raises(EncodingContractError, match="raw-pixel"):
        build_no_resize_image_plan(
            example,
            processor_identity=identity,
            processor_config=_processor_config(max_raw_pixels=10),
        )


def test_merged_visual_token_budget_fails_before_processor_call(tmp_path: Path) -> None:
    example = _raw_example(tmp_path, width=96, height=64)
    identity = _processor_identity()

    with pytest.raises(EncodingContractError, match="merged visual"):
        build_no_resize_image_plan(
            example,
            processor_identity=identity,
            processor_config=_processor_config(max_merged_visual_tokens=1),
        )


def test_processor_grid_mismatch_fails_after_actual_processor_call(tmp_path: Path) -> None:
    example = _raw_example(tmp_path, width=96, height=64)
    components = FakeComponents(
        processor_identity=_processor_identity(),
        processor=FakeProcessor(
            image_grid_thw=torch.tensor([[1, 4, 7]], dtype=torch.long),
            pixel_values=torch.zeros((24, 1536), dtype=torch.float32),
        ),
    )

    with pytest.raises(EncodingContractError, match="image_grid_thw"):
        encode_qwen_image(
            example,
            components=components,
            processor_config=_processor_config(),
        )


def test_do_resize_true_fails_before_processor_call(tmp_path: Path) -> None:
    example = _raw_example(tmp_path, width=96, height=64)

    with pytest.raises(EncodingContractError) as exc_info:
        build_no_resize_image_plan(
            example,
            processor_identity=_processor_identity(),
            processor_config=ProcessorConfig(
                do_resize=True,
                max_raw_pixels=1_000_000,
                max_merged_visual_tokens=4_096,
            ),
        )

    assert exc_info.value.code == "qwen.image_resize_enabled"


def test_decoded_dimension_mismatch_fails_before_processor_call(tmp_path: Path) -> None:
    example = _raw_example(
        tmp_path,
        width=128,
        height=64,
        actual_width=96,
        actual_height=64,
    )
    components = FakeComponents(
        processor_identity=_processor_identity(),
        processor=FakeProcessor(),
    )

    with pytest.raises(EncodingContractError) as exc_info:
        encode_qwen_image(
            example,
            components=components,
            processor_config=_processor_config(),
        )

    assert exc_info.value.code == "qwen.image_dimension_mismatch"


def test_processor_output_missing_grid_or_pixels_fails(tmp_path: Path) -> None:
    example = _raw_example(tmp_path, width=96, height=64)

    missing_grid = FakeComponents(
        processor_identity=_processor_identity(),
        processor=FakeProcessor(include_image_grid_thw=False),
    )
    with pytest.raises(EncodingContractError) as grid_exc:
        encode_qwen_image(
            example,
            components=missing_grid,
            processor_config=_processor_config(),
        )
    assert grid_exc.value.code == "qwen.image_grid_missing"

    missing_pixels = FakeComponents(
        processor_identity=_processor_identity(),
        processor=FakeProcessor(include_pixel_values=False),
    )
    with pytest.raises(EncodingContractError) as pixels_exc:
        encode_qwen_image(
            example,
            components=missing_pixels,
            processor_config=_processor_config(),
        )
    assert pixels_exc.value.code == "qwen.image_pixel_values_missing"


def test_processor_output_bad_shapes_fail(tmp_path: Path) -> None:
    example = _raw_example(tmp_path, width=96, height=64)

    bad_grid_shape = FakeComponents(
        processor_identity=_processor_identity(),
        processor=FakeProcessor(image_grid_thw=torch.tensor([1, 4, 6], dtype=torch.long)),
    )
    with pytest.raises(EncodingContractError) as grid_exc:
        encode_qwen_image(
            example,
            components=bad_grid_shape,
            processor_config=_processor_config(),
        )
    assert grid_exc.value.code == "qwen.image_grid_shape"

    bad_pixel_shape = FakeComponents(
        processor_identity=_processor_identity(),
        processor=FakeProcessor(pixel_values=torch.zeros((23, 1536), dtype=torch.float32)),
    )
    with pytest.raises(EncodingContractError) as pixels_exc:
        encode_qwen_image(
            example,
            components=bad_pixel_shape,
            processor_config=_processor_config(),
        )
    assert pixels_exc.value.code == "qwen.image_pixel_values_shape"


@dataclass(frozen=True)
class FakeComponents:
    processor_identity: QwenProcessorIdentity
    processor: Any


class FakeProcessor:
    def __init__(
        self,
        *,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        include_image_grid_thw: bool = True,
        include_pixel_values: bool = True,
    ) -> None:
        self.image_processor = FakeImageProcessor(
            image_grid_thw=(
                torch.tensor([[1, 4, 6]], dtype=torch.long)
                if image_grid_thw is None
                else image_grid_thw
            ),
            pixel_values=(
                torch.zeros((24, 1536), dtype=torch.float32)
                if pixel_values is None
                else pixel_values
            ),
            include_image_grid_thw=include_image_grid_thw,
            include_pixel_values=include_pixel_values,
        )


class FakeImageProcessor:
    def __init__(
        self,
        *,
        image_grid_thw: torch.Tensor,
        pixel_values: torch.Tensor,
        include_image_grid_thw: bool,
        include_pixel_values: bool,
    ) -> None:
        self.image_grid_thw = image_grid_thw
        self.pixel_values = pixel_values
        self.include_image_grid_thw = include_image_grid_thw
        self.include_pixel_values = include_pixel_values
        self.batch_sizes: list[int] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        images = kwargs.get("images")
        if images is not None:
            self.batch_sizes.append(len(images))
        payload: dict[str, torch.Tensor] = {}
        if self.include_image_grid_thw:
            payload["image_grid_thw"] = self.image_grid_thw
        if self.include_pixel_values:
            payload["pixel_values"] = self.pixel_values
        return payload


def _processor_identity() -> QwenProcessorIdentity:
    return QwenProcessorIdentity(
        processor_class="FakeQwen3VLProcessor",
        tokenizer_class="FakeTokenizer",
        image_processor_class="FakeQwen2VLImageProcessorFast",
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
    )


def _processor_config(
    *,
    max_raw_pixels: int = 1_000_000,
    max_merged_visual_tokens: int = 4_096,
) -> ProcessorConfig:
    return ProcessorConfig(
        do_resize=False,
        max_raw_pixels=max_raw_pixels,
        max_merged_visual_tokens=max_merged_visual_tokens,
    )


def _raw_example(
    tmp_path: Path,
    *,
    width: int,
    height: int,
    actual_width: int | None = None,
    actual_height: int | None = None,
) -> RawExample:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "image.jpg"
    Image.new(
        "RGB",
        (actual_width or width, actual_height or height),
        color=(12, 34, 56),
    ).save(path)
    return RawExample(
        example_id=f"image-{width}x{height}",
        image=ImageRef(
            declared_path=path.name,
            path=path,
            width=width,
            height=height,
            stat={},
        ),
        objects=(RawObject("object-1", "object", (1, 2, 3, 4), {}),),
        metadata={},
        source=SourceProvenance(tmp_path / "examples.jsonl", 1, "abc", "unit"),
    )
