from __future__ import annotations

from pathlib import Path

from src.augmentation.processor import GeometryFlipAugmentationProcessor
from src.config.models import GeometryFlipsAugmentationConfig
from src.data import ImageRef, RawExample, RawObject, SourceProvenance


def test_processor_emits_one_presentation_per_source_example(tmp_path: Path) -> None:
    processor = _processor(horizontal_prob=1.0, vertical_prob=0.0)
    raw = _raw_example(
        tmp_path,
        "ex-a",
        (
            RawObject("a", "left", (100, 200, 300, 400), {}),
        ),
    )

    result = processor.materialize(
        (raw,),
        split="train",
        object_ordering="source_order",
    )

    assert len(result.examples) == 1
    presentation = result.examples[0]
    assert presentation.example_id == "ex-a::aug:hflip"
    assert presentation.objects[0].bbox == (699, 200, 899, 400)
    assert result.receipt["input_example_count"] == 1
    assert result.receipt["output_example_count"] == 1
    assert result.receipt["transform_counts"] == {
        "identity": 0,
        "hflip": 1,
        "vflip": 0,
        "hvflip": 0,
    }


def test_processor_sampling_is_seeded_by_source_identity_not_worker_order(
    tmp_path: Path,
) -> None:
    examples = tuple(
        _raw_example(
            tmp_path / f"ex-{index}",
            f"ex-{index}",
            (RawObject(f"obj-{index}", "object", (10, 20, 30, 40), {}),),
            row_number=index + 1,
        )
        for index in range(12)
    )
    processor_a = _processor(runtime_seed=17, horizontal_prob=0.5, vertical_prob=0.5)
    processor_b = _processor(runtime_seed=17, horizontal_prob=0.5, vertical_prob=0.5)
    processor_c = _processor(runtime_seed=23, horizontal_prob=0.5, vertical_prob=0.5)

    first = processor_a.materialize(
        examples,
        split="train",
        object_ordering="source_order",
    )
    second = processor_b.materialize(
        tuple(reversed(examples)),
        split="train",
        object_ordering="source_order",
    )
    changed_seed = processor_c.materialize(
        examples,
        split="train",
        object_ordering="source_order",
    )

    first_by_source = _transform_by_source_id(first.examples)
    second_by_source = _transform_by_source_id(second.examples)
    changed_by_source = _transform_by_source_id(changed_seed.examples)
    assert first_by_source == second_by_source
    assert first_by_source != changed_by_source


def test_geo_sorted_is_prepared_after_transform(tmp_path: Path) -> None:
    processor = _processor(horizontal_prob=1.0, vertical_prob=0.0)
    raw = _raw_example(
        tmp_path,
        "ex-geo",
        (
            RawObject("left", "left", (100, 100, 200, 200), {}),
            RawObject("right", "right", (700, 100, 800, 200), {}),
        ),
    )

    result = processor.materialize(
        (raw,),
        split="train",
        object_ordering="geo_sorted",
    )

    presentation = result.examples[0]
    assert [obj.object_id for obj in presentation.objects] == ["right", "left"]
    assert [obj.metadata["augmentation"]["original_object_index"] for obj in presentation.objects] == [1, 0]
    assert [obj.bbox for obj in presentation.objects] == [
        (199, 100, 299, 200),
        (799, 100, 899, 200),
    ]


def test_source_order_preserves_tuple_order_after_transform(tmp_path: Path) -> None:
    processor = _processor(horizontal_prob=1.0, vertical_prob=0.0)
    raw = _raw_example(
        tmp_path,
        "ex-source",
        (
            RawObject("left", "left", (100, 100, 200, 200), {}),
            RawObject("right", "right", (700, 100, 800, 200), {}),
        ),
    )

    result = processor.materialize(
        (raw,),
        split="train",
        object_ordering="source_order",
    )

    assert [obj.object_id for obj in result.examples[0].objects] == ["left", "right"]


def test_random_order_records_renderer_seed_key(tmp_path: Path) -> None:
    processor = _processor(runtime_seed=123, horizontal_prob=1.0, vertical_prob=1.0)
    raw = _raw_example(
        tmp_path,
        "ex-random",
        (
            RawObject("a", "a", (100, 100, 200, 200), {}),
            RawObject("b", "b", (700, 100, 800, 200), {}),
        ),
    )

    result = processor.materialize(
        (raw,),
        split="train",
        object_ordering="random",
    )

    presentation = result.examples[0]
    augmentation = presentation.metadata["augmentation"]
    assert presentation.example_id == "ex-random::aug:hvflip"
    assert augmentation["object_order_seed"] == 123
    assert augmentation["object_order_seed_source"] == "123:ex-random::aug:hvflip"
    assert result.receipt["presentation_count"] == 1
    assert result.receipt["random_object_order_presentations"] == [
        {
            "source_example_id": "ex-random",
            "presentation_id": "ex-random::aug:hvflip",
            "object_order_seed": 123,
            "object_order_seed_source": "123:ex-random::aug:hvflip",
        }
    ]


def _processor(
    *,
    runtime_seed: int = 17,
    horizontal_prob: float,
    vertical_prob: float,
) -> GeometryFlipAugmentationProcessor:
    return GeometryFlipAugmentationProcessor(
        config=GeometryFlipsAugmentationConfig(
            enabled=True,
            horizontal_prob=horizontal_prob,
            vertical_prob=vertical_prob,
        ),
        runtime_seed=runtime_seed,
    )


def _raw_example(
    tmp_path: Path,
    example_id: str,
    objects: tuple[RawObject, ...],
    *,
    row_number: int = 1,
) -> RawExample:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake")
    return RawExample(
        example_id=example_id,
        image=ImageRef(
            declared_path="image.jpg",
            path=image_path,
            width=96,
            height=64,
            stat={},
        ),
        objects=objects,
        metadata={},
        source=SourceProvenance(
            tmp_path / "examples.jsonl",
            row_number,
            f"sha-{example_id}",
            "unit",
        ),
    )


def _transform_by_source_id(examples: tuple[RawExample, ...]) -> dict[str, str]:
    return {
        str(example.metadata["augmentation"]["source_example_id"]): str(
            example.metadata["augmentation"]["transform_id"]
        )
        for example in examples
    }
