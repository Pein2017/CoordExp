from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.errors import DataContractError
from src.config import load_train_config
from src.data import ImageRef, RawExample, RawObject, SourceProvenance, load_raw_examples
from src.data.examples import freeze_json
from src.data.geometry import parse_coord_token


FIXTURE = Path("tests/fixtures/smoke/qwen3_vl_single_image_pack")


def test_loads_canonical_smoke_fixture_from_resolved_config() -> None:
    resolved = load_train_config(FIXTURE / "config.yaml")

    examples = load_raw_examples(resolved.config.data.train)

    assert len(examples) == 2
    assert all(isinstance(example, RawExample) for example in examples)
    assert [example.example_id for example in examples] == [
        "coco2017_train_000000000030__smoke2obj",
        "coco2017_train_000000000036__smoke2obj",
    ]
    first = examples[0]
    assert first.image.declared_path == "images/train2017/000000000030.jpg"
    assert first.image.path.is_absolute()
    assert first.image.path.exists()
    assert (first.image.width, first.image.height) == (1248, 832)
    assert tuple(obj.object_id for obj in first.objects) == ("291613", "1155486")
    assert first.objects[0].bbox == (319, 72, 718, 830)
    assert first.objects[0].description == "potted plant"
    assert first.source.row_number == 1
    assert first.source.source_format == "canonical_raw_example"
    assert isinstance(first.objects, tuple)


def test_loads_current_len12000_coord_jsonl_shape(tmp_path: Path) -> None:
    image = tmp_path / "images" / "train2017" / "000000000030.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-jpg-bytes")
    row = _current_source_row(image_ref="images/train2017/000000000030.jpg")
    jsonl = _write_jsonl(tmp_path, row)

    (example,) = load_raw_examples(jsonl)

    assert example.example_id == "coco2017_train_000000000030"
    assert example.image.path == image.resolve()
    assert example.objects[0].object_id == "291613"
    assert example.objects[0].bbox == (319, 72, 718, 830)
    assert example.metadata["source"]["file_name"] == "images/train2017/000000000030.jpg"
    assert example.objects[0].metadata["source"]["category_name"] == "potted plant"
    assert example.source.source_format == "coord_jsonl_len12000"


def test_current_len12000_source_accepts_approved_sibling_image_reference(tmp_path: Path) -> None:
    source_root = tmp_path / "rescale_32_1024_bbox_len12000"
    image = tmp_path / "rescale_32_1024_bbox" / "images" / "train2017" / "000000000030.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-jpg-bytes")
    row = _current_source_row(
        image_ref="../rescale_32_1024_bbox/images/train2017/000000000030.jpg",
    )

    (example,) = load_raw_examples(_write_jsonl(source_root, row))

    assert example.image.path == image.resolve()
    assert example.metadata["source"]["original_images"] == (
        "../rescale_32_1024_bbox/images/train2017/000000000030.jpg",
    )


def test_sample_limit_counts_accepted_valid_examples(tmp_path: Path) -> None:
    first = _canonical_row(tmp_path, example_id="a", object_id="o1")
    invalid_second = _canonical_row(tmp_path, example_id="b", object_id="o2")
    invalid_second["objects"][0]["bbox"] = [1, 2, 3]
    jsonl = _write_jsonl(tmp_path, first, invalid_second)

    examples = load_raw_examples(jsonl, sample_limit=1)

    assert [example.example_id for example in examples] == ["a"]


def test_invalid_rows_before_sample_limit_still_fail(tmp_path: Path) -> None:
    invalid_first = _canonical_row(tmp_path, example_id="a", object_id="o1")
    invalid_first["objects"][0]["bbox"] = [1, 2, 3]
    second = _canonical_row(tmp_path, example_id="b", object_id="o2")

    with pytest.raises(DataContractError) as exc_info:
        load_raw_examples(_write_jsonl(tmp_path, invalid_first, second), sample_limit=1)

    assert exc_info.value.code == "data.bbox_shape"


def test_sample_limit_rejects_boolean_direct_api_value(tmp_path: Path) -> None:
    with pytest.raises(DataContractError) as exc_info:
        load_raw_examples(_write_jsonl(tmp_path, _canonical_row(tmp_path)), sample_limit=True)

    assert exc_info.value.code == "data.sample_limit"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda row: row.update({"extra": True}), "data.unknown_fields"),
        (lambda row: row.update({"video": "clip.mp4"}), "data.video_unsupported"),
        (lambda row: row["image"].update({"path": "missing.jpg"}), "data.image_missing"),
        (lambda row: row["image"].update({"width": 0}), "data.image_dimension_range"),
        (lambda row: row["objects"].clear(), "data.objects_empty"),
        (lambda row: row["objects"][0].pop("object_id"), "data.object_id_missing"),
        (lambda row: row["objects"][0].update({"object_id": ""}), "data.string_empty"),
        (lambda row: row["objects"][0].update({"description": "bad\tthing"}), "data.description_control_whitespace"),
        (lambda row: row["objects"][0].update({"bbox": [1, 2, 3]}), "data.bbox_shape"),
        (lambda row: row["objects"][0].update({"bbox": [1, 2, 1000, 4]}), "data.bbox_value_range"),
        (lambda row: row["objects"][0].update({"bbox": [4, 2, 4, 5]}), "data.bbox_order"),
    ],
)
def test_canonical_shape_failures(tmp_path: Path, mutate, code: str) -> None:
    row = _canonical_row(tmp_path)
    mutate(row)
    jsonl = _write_jsonl(tmp_path, row)

    with pytest.raises(DataContractError) as exc_info:
        load_raw_examples(jsonl)

    assert exc_info.value.code == code


def test_canonical_image_path_must_stay_under_jsonl_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    external_image = tmp_path / "external" / "image.jpg"
    external_image.parent.mkdir(parents=True)
    external_image.write_bytes(b"bytes")
    row = _canonical_row(dataset_root)
    row["image"]["path"] = str(external_image)

    with pytest.raises(DataContractError) as exc_info:
        load_raw_examples(_write_jsonl(dataset_root, row))

    assert exc_info.value.code == "data.image_path_escape"


def test_duplicate_example_and_object_ids_fail(tmp_path: Path) -> None:
    first = _canonical_row(tmp_path, example_id="dup", object_id="o1")
    second = _canonical_row(tmp_path, example_id="dup", object_id="o2")

    with pytest.raises(DataContractError) as exc_info:
        load_raw_examples(_write_jsonl(tmp_path, first, second))

    assert exc_info.value.code == "data.duplicate_example_id"

    row = _canonical_row(tmp_path)
    row["objects"].append(dict(row["objects"][0]))
    with pytest.raises(DataContractError) as object_exc:
        load_raw_examples(_write_jsonl(tmp_path, row))

    assert object_exc.value.code == "data.duplicate_object_id"


def test_direct_record_construction_freezes_canonical_state(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"bytes")
    object_metadata = {"source": {"category_name": "thing"}}
    raw_object = RawObject("object-1", "thing", [1, 2, 3, 4], object_metadata)
    objects = [raw_object]
    example_metadata = {"source": {"dataset": "toy"}}

    example = RawExample(
        "example-1",
        ImageRef("image.jpg", image_path, 64, 32, {"path": str(image_path)}),
        objects,
        example_metadata,
        SourceProvenance(tmp_path / "examples.jsonl", 1, "abc", "manual"),
    )
    objects.append(RawObject("object-2", "other", [5, 6, 7, 8], {}))
    object_metadata["source"]["category_name"] = "changed"
    example_metadata["source"]["dataset"] = "changed"

    assert isinstance(example.objects, tuple)
    assert len(example.objects) == 1
    assert example.objects[0].bbox == (1, 2, 3, 4)
    assert example.objects[0].metadata["source"]["category_name"] == "thing"
    assert example.metadata["source"]["dataset"] == "toy"
    with pytest.raises(TypeError):
        example.metadata["new"] = "blocked"


def test_direct_record_construction_rejects_duplicate_object_ids(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"bytes")
    duplicate_objects = (
        RawObject("same", "first", [1, 2, 3, 4], {}),
        RawObject("same", "second", [5, 6, 7, 8], {}),
    )

    with pytest.raises(DataContractError) as exc_info:
        RawExample(
            "example-1",
            ImageRef("image.jpg", image_path, 64, 32, {"path": str(image_path)}),
            duplicate_objects,
            {},
            SourceProvenance(tmp_path / "examples.jsonl", 1, "abc", "manual"),
        )

    assert exc_info.value.code == "data.duplicate_object_id"


@pytest.mark.parametrize(
    "token",
    [
        "<|coord_001|>",
        "<|coord_1000|>",
        " <|coord_1|>",
        "<|coord_1|> ",
        "1",
        1,
        "<|object_ref_start|>",
    ],
)
def test_coordinate_token_parser_is_strict(token) -> None:
    with pytest.raises(DataContractError):
        parse_coord_token(token, field="bbox_2d[0]")


def test_current_source_rejects_multi_image_and_loose_bbox_token(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"bytes")
    row = {
        "image_id": 1,
        "file_name": "image.jpg",
        "images": ["image.jpg", "image.jpg"],
        "width": 10,
        "height": 10,
        "metadata": {"source": "toy", "split": "train"},
        "objects": [
            {
                "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
                "desc": "thing",
                "category_id": 1,
                "category_name": "thing",
                "coco_ann_id": 1,
            }
        ],
    }

    with pytest.raises(DataContractError) as image_exc:
        load_raw_examples(_write_jsonl(tmp_path, row))

    assert image_exc.value.code == "data.image_count"

    row["images"] = ["image.jpg"]
    row["objects"][0]["bbox_2d"][0] = "<|coord_001|>"
    with pytest.raises(DataContractError) as token_exc:
        load_raw_examples(_write_jsonl(tmp_path, row))

    assert token_exc.value.code == "data.coord_token_format"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda row: row.pop("file_name"), "data.missing_fields"),
        (lambda row: row.update({"metadata": "bad"}), "data.source_metadata_shape"),
        (lambda row: row["metadata"].pop("source"), "data.string_type"),
        (lambda row: row.update({"image_id": 30.7}), "data.example_id_type"),
        (lambda row: row.update({"file_name": "images/train2017/other.jpg"}), "data.source_image_mismatch"),
        (lambda row: row["objects"][0].update({"bbox_2d": None}), "data.bbox_shape"),
        (lambda row: row["objects"][0].update({"coco_ann_id": True}), "data.object_id_type"),
    ],
)
def test_current_source_required_identity_and_provenance_failures(
    tmp_path: Path,
    mutate,
    code: str,
) -> None:
    image = tmp_path / "images" / "train2017" / "000000000030.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-jpg-bytes")
    row = _current_source_row(image_ref="images/train2017/000000000030.jpg")
    mutate(row)

    with pytest.raises(DataContractError) as exc_info:
        load_raw_examples(_write_jsonl(tmp_path, row))

    assert exc_info.value.code == code


def test_jsonl_rejects_nonstandard_json_constants(tmp_path: Path) -> None:
    row = _canonical_row(tmp_path)
    row["metadata"]["source"]["score"] = float("nan")
    jsonl = _write_jsonl(tmp_path, row)

    with pytest.raises(DataContractError) as exc_info:
        load_raw_examples(jsonl)

    assert exc_info.value.code == "data.json_constant"


def test_freeze_json_rejects_nonfinite_python_floats() -> None:
    with pytest.raises(DataContractError) as exc_info:
        freeze_json({"bad": float("inf")})

    assert exc_info.value.code == "data.json_nonfinite_number"


def _canonical_row(
    tmp_path: Path,
    *,
    example_id: str = "example-1",
    object_id: str = "object-1",
) -> dict:
    image = tmp_path / "image.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"bytes")
    return {
        "example_id": example_id,
        "image": {"path": "image.jpg", "width": 64, "height": 32},
        "objects": [
            {
                "object_id": object_id,
                "description": "thing",
                "bbox": [1, 2, 3, 4],
                "metadata": {"source": {"category_name": "thing"}},
            }
        ],
        "metadata": {"source": {"dataset": "toy"}},
    }


def _current_source_row(*, image_ref: str) -> dict:
    return {
        "image_id": 30,
        "file_name": "images/train2017/000000000030.jpg",
        "images": [image_ref],
        "width": 1248,
        "height": 832,
        "metadata": {"source": "coco2017", "split": "train"},
        "objects": [
            {
                "bbox_2d": [
                    "<|coord_319|>",
                    "<|coord_72|>",
                    "<|coord_718|>",
                    "<|coord_830|>",
                ],
                "desc": "potted plant",
                "category_id": 64,
                "category_name": "potted plant",
                "coco_ann_id": 291613,
            }
        ],
    }


def _write_jsonl(tmp_path: Path, *rows: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "examples.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
