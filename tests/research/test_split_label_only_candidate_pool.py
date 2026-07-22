from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.split_label_only_candidate_pool import (
    split_label_only_candidate_pool,
)


def _raw_row(image_id: int, object_count: int) -> bytes:
    objects = [{"index": index} for index in range(object_count)]
    return (
        '{ "image_id" : %d, "objects" : %s, "note" : "row-%d" }\n'
        % (image_id, json.dumps(objects, separators=(",", ":")), image_id)
    ).encode("utf-8")


def _write_balanced_pool(path: Path) -> list[bytes]:
    rows: list[bytes] = []
    for image_id, object_count in enumerate((1, 4, 8, 16) * 3, start=100):
        rows.append(_raw_row(image_id, object_count))
    path.write_bytes(b"".join(rows))
    return rows


def _write_balanced_pool_with_per_band_count(path: Path, per_band_count: int) -> None:
    rows: list[bytes] = []
    for repeat in range(per_band_count):
        for offset, object_count in enumerate((1, 4, 8, 16)):
            rows.append(_raw_row(10_000 + repeat * 4 + offset, object_count))
    path.write_bytes(b"".join(rows))


def _read_jsonl_bytes(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines(keepends=True)


def test_splits_byte_preserving_balanced_rows_in_source_order_and_writes_receipt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate-pool.jsonl"
    raw_rows = _write_balanced_pool(source)
    output_dir = tmp_path / "split"

    receipt = split_label_only_candidate_pool(
        input_path=source,
        output_dir=output_dir,
        train_count=4,
        development_count=4,
        heldout_count=4,
        seed=41,
    )

    expected_rows = set(raw_rows)
    seen_rows: list[bytes] = []
    for split_name, filename in (
        ("train_candidate", "train-candidate.jsonl"),
        ("development", "development.jsonl"),
        ("heldout", "heldout.jsonl"),
    ):
        split_rows = _read_jsonl_bytes(output_dir / filename)
        seen_rows.extend(split_rows)
        assert all(row in expected_rows for row in split_rows)
        source_positions = [raw_rows.index(row) for row in split_rows]
        assert source_positions == sorted(source_positions)
        assert receipt["outputs"][split_name]["count"] == 4
        assert receipt["outputs"][split_name]["per_band_counts"] == {
            "sparse_1_to_3": 1,
            "medium_4_to_7": 1,
            "dense_8_to_15": 1,
            "very_dense_16_plus": 1,
        }

    assert len(seen_rows) == len(set(seen_rows)) == 12
    assert receipt["proof"]["zero_overlap"] is True
    assert receipt["proof"]["union_matches_selected"] is True
    assert receipt["proof"]["union_matches_input"] is True
    assert receipt["proof"]["unselected_input_count"] == 0
    assert receipt["input"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    persisted = json.loads((output_dir / "split-receipt.json").read_text(encoding="utf-8"))
    assert persisted == receipt


def test_same_seed_has_same_split_contents(tmp_path: Path) -> None:
    source = tmp_path / "candidate-pool.jsonl"
    _write_balanced_pool(source)
    first = tmp_path / "first"
    second = tmp_path / "second"

    split_label_only_candidate_pool(
        input_path=source,
        output_dir=first,
        train_count=4,
        development_count=4,
        heldout_count=4,
        seed=7,
    )
    split_label_only_candidate_pool(
        input_path=source,
        output_dir=second,
        train_count=4,
        development_count=4,
        heldout_count=4,
        seed=7,
    )

    for filename in ("train-candidate.jsonl", "development.jsonl", "heldout.jsonl"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_defaults_materialize_the_intended_2048_256_128_shape(tmp_path: Path) -> None:
    source = tmp_path / "candidate-pool-2432.jsonl"
    _write_balanced_pool_with_per_band_count(source, per_band_count=608)

    receipt = split_label_only_candidate_pool(
        input_path=source,
        output_dir=tmp_path / "split",
    )

    assert receipt["requested_counts"] == {
        "train_candidate": 2048,
        "development": 256,
        "heldout": 128,
    }
    assert receipt["requested_per_band_counts"] == {
        "train_candidate": {
            "sparse_1_to_3": 512,
            "medium_4_to_7": 512,
            "dense_8_to_15": 512,
            "very_dense_16_plus": 512,
        },
        "development": {
            "sparse_1_to_3": 64,
            "medium_4_to_7": 64,
            "dense_8_to_15": 64,
            "very_dense_16_plus": 64,
        },
        "heldout": {
            "sparse_1_to_3": 32,
            "medium_4_to_7": 32,
            "dense_8_to_15": 32,
            "very_dense_16_plus": 32,
        },
    }
    assert receipt["proof"]["selected_image_count"] == 2432
    assert receipt["proof"]["union_image_count"] == 2432
    assert receipt["proof"]["unselected_input_count"] == 0


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ([_raw_row(1, 1), _raw_row(1, 4)], "duplicate image_id"),
        ([b'{"image_id":1,"objects":[]}\n'], "empty objects"),
    ],
)
def test_rejects_invalid_source_rows(
    tmp_path: Path, rows: list[bytes], error: str
) -> None:
    source = tmp_path / "candidate-pool.jsonl"
    source.write_bytes(b"".join(rows))

    with pytest.raises(ValueError, match=error):
        split_label_only_candidate_pool(
            input_path=source,
            output_dir=tmp_path / "split",
            train_count=0,
            development_count=0,
            heldout_count=0,
        )


def test_rejects_existing_immutable_output(tmp_path: Path) -> None:
    source = tmp_path / "candidate-pool.jsonl"
    _write_balanced_pool(source)
    output_dir = tmp_path / "split"
    output_dir.mkdir()
    (output_dir / "development.jsonl").write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable output already exists"):
        split_label_only_candidate_pool(
            input_path=source,
            output_dir=output_dir,
            train_count=4,
            development_count=4,
            heldout_count=4,
        )


def test_rejects_requested_count_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "candidate-pool.jsonl"
    _write_balanced_pool(source)

    with pytest.raises(ValueError, match="requested split count total 11"):
        split_label_only_candidate_pool(
            input_path=source,
            output_dir=tmp_path / "split",
            train_count=4,
            development_count=4,
            heldout_count=3,
        )


def test_relative_image_paths_require_outputs_beside_source(tmp_path: Path) -> None:
    source = tmp_path / "candidate-pool.jsonl"
    source.write_text(
        json.dumps(
            {
                "image_id": 1,
                "images": ["../../images/one.jpg"],
                "objects": [{"index": 0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be written beside"):
        split_label_only_candidate_pool(
            input_path=source,
            output_dir=tmp_path / "nested",
            train_count=1,
            development_count=0,
            heldout_count=0,
        )
