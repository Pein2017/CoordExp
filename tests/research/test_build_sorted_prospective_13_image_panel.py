"""CPU contract tests for the prospective 13-image panel-admission probe."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pytest

from scripts.research import build_sorted_prospective_13_image_panel as builder


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row(image_id: int, *, images: list[str], objects: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": images,
        "objects": objects,
        "width": 100,
        "height": 100,
        "image_id": image_id,
        "file_name": f"images/val2017/{image_id:012d}.jpg",
        "metadata": {"source": "coco2017", "split": "val"},
    }


def _owner(desc: str, ann_id: int) -> dict[str, object]:
    return {
        "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
        "desc": desc,
        "category_id": 1 if desc == "person" else 32,
        "category_name": desc,
        "coco_ann_id": ann_id,
    }


def _make_fixture(tmp_path: Path) -> tuple[builder.SourcePaths, Path]:
    """Mirrors the real two-tier path convention: an authority file one level

    below its root (like ``public_data/coco/rescale_32_1024_bbox_len12000/``)
    and panel files two levels below the same root inside an
    ``evaluation-inputs/`` directory (like
    ``outputs/.../<unit>/evaluation-inputs/``), both referencing one shared
    ``images/val2017/`` tree.
    """

    images_dir = tmp_path / "images" / "val2017"
    images_dir.mkdir(parents=True)
    target_image = images_dir / "000000002299.jpg"
    target_image.write_bytes(b"fake-jpeg-bytes-2299")
    legacy_image_1584 = images_dir / "000000001584.jpg"
    legacy_image_1584.write_bytes(b"fake-jpeg-bytes-1584")
    legacy_image_2685 = images_dir / "000000002685.jpg"
    legacy_image_2685.write_bytes(b"fake-jpeg-bytes-2685")

    legacy_dir = tmp_path / "legacy" / "evaluation-inputs"
    legacy_dir.mkdir(parents=True)
    legacy_rows = [
        _row(1584, images=["../../images/val2017/000000001584.jpg"], objects=[_owner("person", -1)]),
        _row(2685, images=["../../images/val2017/000000002685.jpg"], objects=[_owner("person", -2)]),
    ]
    legacy_lines = [
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in legacy_rows
    ]
    legacy_panel = legacy_dir / "human-refined-12.coord.jsonl"
    legacy_panel.write_bytes(b"".join(legacy_lines))

    authority_dir = tmp_path / "authority"
    authority_dir.mkdir(parents=True)
    authority_path = authority_dir / "val.coord.jsonl"
    target_objects = [_owner("person", index) for index in range(3)] + [
        _owner("tie", index) for index in range(2)
    ]
    target_row = _row(
        2299, images=["../images/val2017/000000002299.jpg"], objects=target_objects
    )
    target_line = (json.dumps(target_row, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    decoy_row = _row(9999, images=["../images/val2017/000000002299.jpg"], objects=[_owner("person", 9)])
    decoy_line = (json.dumps(decoy_row, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    authority_path.write_bytes(decoy_line + target_line)

    sources = builder.SourcePaths(
        legacy_panel=legacy_panel,
        legacy_panel_sha256=_sha256(legacy_panel.read_bytes()),
        expected_legacy_line_count=2,
        authority=authority_path,
        authority_sha256=_sha256(authority_path.read_bytes()),
        target_image_id=2299,
        target_line_sha256=_sha256(target_line),
        target_image_sha256=_sha256(target_image.read_bytes()),
        expected_owner_total=5,
        expected_person_count=3,
        expected_tie_count=2,
        expected_legacy_owner_total=2,
        expected_full_panel_owner_total=7,
    )
    return sources, target_image


def test_inserts_target_row_in_numeric_order_and_preserves_legacy_bytes(
    tmp_path: Path,
) -> None:
    sources, target_image = _make_fixture(tmp_path)
    output_root = tmp_path / "unit"

    result = builder.build_panel(output_root, sources=sources)

    assert result["status"] == "created"
    assert result["output"]["ordered_image_ids"] == [1584, 2299, 2685]
    assert result["output"]["insertion_index"] == 1
    assert result["output"]["legacy_lines_preserved_byte_for_byte"] is True
    assert result["legacy_owner_counts"] == {"total": 2, "by_desc": {"person": 2}}
    assert result["target_owner_counts"] == {
        "total": 5,
        "person": 3,
        "tie": 2,
        "by_desc": {"person": 3, "tie": 2},
    }
    assert result["full_panel_owner_counts"] == {"total": 7, "by_desc": {"person": 5, "tie": 2}}

    jsonl_path = output_root / "evaluation-inputs" / "human-refined-13.coord.jsonl"
    lines = jsonl_path.read_bytes().splitlines(keepends=True)
    assert len(lines) == 3
    legacy_lines = sources.legacy_panel.read_bytes().splitlines(keepends=True)
    assert lines[0] == legacy_lines[0]
    assert lines[2] == legacy_lines[1]

    inserted_row = json.loads(lines[1])
    assert inserted_row["image_id"] == 2299
    assert inserted_row["file_name"] == "images/val2017/000000002299.jpg"
    assert inserted_row["objects"] == json.loads(
        (tmp_path / "authority" / "val.coord.jsonl").read_bytes().splitlines()[1]
    )["objects"]
    resolved = (jsonl_path.parent / inserted_row["images"][0]).resolve()
    assert resolved == target_image.resolve()

    manifest = result["images_manifest"]
    assert manifest["sha256"] == builder.sha256_json(manifest["entries"])
    assert [entry["image_id"] for entry in manifest["entries"]] == [1584, 2299, 2685]
    for entry in manifest["entries"]:
        assert Path(entry["resolved_path"]).is_file()
        assert entry["byte_size"] == Path(entry["resolved_path"]).stat().st_size
        assert entry["sha256"] == _sha256(Path(entry["resolved_path"]).read_bytes())
    target_entry = next(e for e in manifest["entries"] if e["image_id"] == 2299)
    assert target_entry["sha256"] == sources.target_image_sha256
    assert target_entry["resolved_path"] == str(target_image.resolve())

    receipt_path = output_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_receipt = {
        key: value for key, value in result.items() if key not in {"status", "output_root"}
    }
    assert receipt == expected_receipt
    assert receipt["receipt_content_sha256"] == builder.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )


def test_rerun_is_a_no_op_identical_publish(tmp_path: Path) -> None:
    sources, _ = _make_fixture(tmp_path)
    output_root = tmp_path / "unit"

    first = builder.build_panel(output_root, sources=sources)
    assert first["status"] == "created"
    jsonl_path = output_root / "evaluation-inputs" / "human-refined-13.coord.jsonl"
    receipt_path = output_root / "receipt.json"
    jsonl_before = jsonl_path.read_bytes()
    receipt_before = receipt_path.read_bytes()

    second = builder.build_panel(output_root, sources=sources)
    assert second["status"] == "identical_existing_output"
    assert jsonl_path.read_bytes() == jsonl_before
    assert receipt_path.read_bytes() == receipt_before
    assert {k: v for k, v in first.items() if k != "status"} == {
        k: v for k, v in second.items() if k != "status"
    }


def test_rejects_mismatched_existing_output(tmp_path: Path) -> None:
    sources, _ = _make_fixture(tmp_path)
    output_root = tmp_path / "unit"
    builder.build_panel(output_root, sources=sources)

    (output_root / "receipt.json").write_bytes(b'{"tampered":true}\n')

    with pytest.raises(builder.PanelContractError, match="non-identical content"):
        builder.build_panel(output_root, sources=sources)


def test_rejects_foreign_file_in_output_root(tmp_path: Path) -> None:
    sources, _ = _make_fixture(tmp_path)
    output_root = tmp_path / "unit"
    builder.build_panel(output_root, sources=sources)

    (output_root / "stray.txt").write_text("not part of this panel", encoding="utf-8")

    with pytest.raises(builder.PanelContractError, match="foreign or partial file set"):
        builder.build_panel(output_root, sources=sources)


def test_rejects_broken_legacy_image_path(tmp_path: Path) -> None:
    sources, _ = _make_fixture(tmp_path)
    (tmp_path / "images" / "val2017" / "000000001584.jpg").unlink()

    with pytest.raises(
        builder.PanelContractError,
        match=r"panel row 1 \(image_id=1584\) image reference does not resolve",
    ):
        builder.build_panel(tmp_path / "unit", sources=sources)

    assert not (tmp_path / "unit").exists()


@pytest.mark.parametrize(
    "mutate,match",
    [
        ("legacy_panel_sha256", "legacy panel sha256 mismatch"),
        ("expected_legacy_line_count", "legacy panel line count mismatch"),
        ("authority_sha256", "authority file sha256 mismatch"),
        ("target_line_sha256", "target authority line sha256 mismatch"),
        ("target_image_sha256", "target image sha256 mismatch"),
        ("expected_owner_total", "owner-count mismatch"),
        ("expected_legacy_owner_total", "legacy panel object-count mismatch"),
        ("expected_full_panel_owner_total", "full-panel object-count mismatch"),
    ],
)
def test_rejects_hash_and_count_mismatches(
    tmp_path: Path, mutate: str, match: str
) -> None:
    sources, _ = _make_fixture(tmp_path)
    bad_value: object
    if mutate == "expected_legacy_line_count":
        bad_value = 3
    elif mutate in {"expected_owner_total", "expected_legacy_owner_total", "expected_full_panel_owner_total"}:
        bad_value = 999
    else:
        bad_value = "0" * 64
    bad_sources = replace(sources, **{mutate: bad_value})

    with pytest.raises(builder.PanelContractError, match=match):
        builder.build_panel(tmp_path / "unit", sources=bad_sources)


def test_rejects_full_panel_count_drift(tmp_path: Path) -> None:
    """Full-panel total must equal expected 392-style total *and* legacy+target."""

    sources, _ = _make_fixture(tmp_path)

    drifted_expectation = replace(sources, expected_full_panel_owner_total=999)
    with pytest.raises(builder.PanelContractError, match="full-panel object-count mismatch"):
        builder.build_panel(tmp_path / "unit-drift-expectation", sources=drifted_expectation)

    drifted_legacy = replace(
        sources, expected_legacy_owner_total=1, expected_full_panel_owner_total=6
    )
    with pytest.raises(builder.PanelContractError, match="legacy panel object-count mismatch"):
        builder.build_panel(tmp_path / "unit-drift-legacy", sources=drifted_legacy)


def test_rejects_duplicate_target_image_id_in_legacy_panel(tmp_path: Path) -> None:
    sources, _ = _make_fixture(tmp_path)
    duplicated_sources = replace(sources, target_image_id=1584)

    with pytest.raises(builder.PanelContractError, match="already present in the legacy panel"):
        builder.build_panel(tmp_path / "unit", sources=duplicated_sources)


def test_rejects_zero_or_multiple_authority_matches(tmp_path: Path) -> None:
    sources, _ = _make_fixture(tmp_path)

    missing_sources = replace(sources, target_image_id=13579)
    with pytest.raises(builder.PanelContractError, match="found 0"):
        builder.build_panel(tmp_path / "unit-missing", sources=missing_sources)

    authority_path = sources.authority
    authority_lines = authority_path.read_bytes().splitlines(keepends=True)
    duplicated_authority = authority_path.parent / "val-duplicated.coord.jsonl"
    duplicated_authority.write_bytes(authority_lines[1] + authority_lines[1])
    duplicated_sources = replace(
        sources,
        authority=duplicated_authority,
        authority_sha256=_sha256(duplicated_authority.read_bytes()),
    )
    with pytest.raises(builder.PanelContractError, match="found 2"):
        builder.build_panel(tmp_path / "unit-duplicated", sources=duplicated_sources)


def test_rejects_unsorted_or_duplicate_legacy_image_ids(tmp_path: Path) -> None:
    sources, _ = _make_fixture(tmp_path)

    unsorted_rows = [
        _row(2685, images=["../../images/val2017/000000002685.jpg"], objects=[_owner("person", -1)]),
        _row(1584, images=["../../images/val2017/000000001584.jpg"], objects=[_owner("person", -2)]),
    ]
    unsorted_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in unsorted_rows
    )
    unsorted_panel = sources.legacy_panel.parent / "unsorted.jsonl"
    unsorted_panel.write_bytes(unsorted_bytes)
    unsorted_sources = replace(
        sources,
        legacy_panel=unsorted_panel,
        legacy_panel_sha256=_sha256(unsorted_bytes),
    )
    with pytest.raises(builder.PanelContractError, match="not sorted ascending"):
        builder.build_panel(tmp_path / "unit-unsorted", sources=unsorted_sources)

    duplicate_rows = [
        _row(1584, images=["../../images/val2017/000000001584.jpg"], objects=[_owner("person", -1)]),
        _row(1584, images=["../../images/val2017/000000001584.jpg"], objects=[_owner("person", -2)]),
    ]
    duplicate_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in duplicate_rows
    )
    duplicate_panel = sources.legacy_panel.parent / "duplicate.jsonl"
    duplicate_panel.write_bytes(duplicate_bytes)
    duplicate_sources = replace(
        sources,
        legacy_panel=duplicate_panel,
        legacy_panel_sha256=_sha256(duplicate_bytes),
    )
    with pytest.raises(builder.PanelContractError, match="duplicate image_id"):
        builder.build_panel(tmp_path / "unit-duplicate", sources=duplicate_sources)


def test_rejects_target_row_with_more_than_one_image(tmp_path: Path) -> None:
    sources, _ = _make_fixture(tmp_path)
    authority_dir = sources.authority.parent
    target_row = _row(
        2299,
        images=["../images/val2017/000000002299.jpg", "../images/val2017/000000002299.jpg"],
        objects=[_owner("person", index) for index in range(3)]
        + [_owner("tie", index) for index in range(2)],
    )
    line = (json.dumps(target_row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    multi_image_authority = authority_dir / "val-multi-image.coord.jsonl"
    multi_image_authority.write_bytes(line)
    multi_image_sources = replace(
        sources,
        authority=multi_image_authority,
        authority_sha256=_sha256(line),
        target_line_sha256=_sha256(line),
    )
    with pytest.raises(builder.PanelContractError, match="exactly one image"):
        builder.build_panel(tmp_path / "unit-multi-image", sources=multi_image_sources)


def test_real_default_sources_end_to_end() -> None:
    if not builder.DEFAULT_SOURCES.legacy_panel.exists() or not builder.DEFAULT_SOURCES.authority.exists():
        pytest.skip("real CoordExp research data root is not available in this environment")

    # The legacy panel's byte-preserved image references are relative to a
    # fixed real-repository depth (outputs/research/qwen3-vl-dense-enumeration/
    # <unit>/evaluation-inputs/), so the throwaway probe root must live at
    # that same depth for full-panel path resolution to succeed.
    real_check_root = builder.OUTPUTS / f".pytest-real-check-{uuid.uuid4().hex[:12]}"
    assert not real_check_root.exists()
    try:
        result = builder.build_panel(real_check_root)

        assert result["status"] == "created"
        assert result["output"]["line_count"] == 13
        assert result["output"]["ordered_image_ids"] == [
            1584, 2299, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228,
        ]
        assert result["output"]["insertion_index"] == 1
        assert result["legacy_owner_counts"]["total"] == 346
        assert result["target_owner_counts"] == {
            "total": 46,
            "person": 38,
            "tie": 8,
            "by_desc": {"person": 38, "tie": 8},
        }
        assert result["full_panel_owner_counts"]["total"] == 392
        assert result["inputs"]["target_image"]["sha256"] == builder.TARGET_IMAGE_SHA256

        manifest = result["images_manifest"]
        assert len(manifest["entries"]) == 13
        assert [entry["image_id"] for entry in manifest["entries"]] == result["output"][
            "ordered_image_ids"
        ]
        for entry in manifest["entries"]:
            resolved = Path(entry["resolved_path"])
            assert resolved.is_file()
            assert entry["byte_size"] == resolved.stat().st_size
            assert entry["sha256"] == builder.sha256_file(resolved)
        target_entry = next(e for e in manifest["entries"] if e["image_id"] == 2299)
        assert target_entry["sha256"] == builder.TARGET_IMAGE_SHA256
        assert target_entry["resolved_path"] == str(
            builder.DEFAULT_SOURCES.authority.parent.parent
            / "rescale_32_1024_bbox/images/val2017/000000002299.jpg"
        )

        jsonl_path = real_check_root / "evaluation-inputs" / "human-refined-13.coord.jsonl"
        lines = jsonl_path.read_bytes().splitlines(keepends=True)
        legacy_lines = builder.DEFAULT_SOURCES.legacy_panel.read_bytes().splitlines(keepends=True)
        assert lines[0] == legacy_lines[0]
        assert lines[2:] == legacy_lines[1:]

        second = builder.build_panel(real_check_root)
        assert second["status"] == "identical_existing_output"
    finally:
        shutil.rmtree(real_check_root, ignore_errors=True)
