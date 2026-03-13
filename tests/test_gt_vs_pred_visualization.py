from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.infer.vis import render_vis_from_jsonl
from src.vis import compose_comparison_scenes_from_jsonls
from src.vis.gt_vs_pred import (
    materialize_eval_gt_vs_pred_vis_resource,
    materialize_gt_vs_pred_vis_resource,
    render_gt_vs_pred_review,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _bbox(desc: str, points: list[int]) -> dict:
    return {"type": "bbox_2d", "points": points, "desc": desc}


def test_materialize_vis_resource_canonicalizes_gt_and_preserves_pred_order(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "image": "img.png",
                "width": 100,
                "height": 100,
                "coord_mode": "pixel",
                "gt": [
                    _bbox("zebra", [40, 40, 60, 60]),
                    _bbox("apple", [10, 10, 20, 20]),
                ],
                "pred": [
                    _bbox("pred-first", [40, 40, 60, 60]),
                    _bbox("pred-second", [10, 10, 20, 20]),
                ],
            }
        ],
    )

    out_path = materialize_gt_vs_pred_vis_resource(source_path)
    rows = _read_jsonl(out_path)
    assert len(rows) == 1

    row = rows[0]
    assert row["schema_version"] == 1
    assert row["coord_mode"] == "pixel"
    assert [obj["desc"] for obj in row["gt"]] == ["apple", "zebra"]
    assert [obj["index"] for obj in row["gt"]] == [0, 1]
    assert [obj["desc"] for obj in row["pred"]] == ["pred-first", "pred-second"]
    assert [obj["index"] for obj in row["pred"]] == [0, 1]
    assert row["matching"]["pred_index_domain"] == "canonical_pred_index"
    assert row["matching"]["gt_index_domain"] == "canonical_gt_index"
    assert row["matching"]["matched_pairs"] == [[0, 1], [1, 0]]


def test_materialize_vis_resource_normalizes_external_eval_matches(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "image": "img.png",
                "width": 100,
                "height": 100,
                "coord_mode": "pixel",
                "gt": [
                    _bbox("zebra", [40, 40, 60, 60]),
                    _bbox("apple", [10, 10, 20, 20]),
                ],
                "pred": [
                    _bbox("pred-first", [40, 40, 60, 60]),
                    _bbox("pred-second", [10, 10, 20, 20]),
                ],
            }
        ],
    )

    out_path = materialize_gt_vs_pred_vis_resource(
        source_path,
        external_matches={
            0: {
                "image_id": 0,
                "matches": [{"pred_idx": 0, "gt_idx": 0}],
                "unmatched_gt_indices": [1],
                "unmatched_pred_indices": [1],
                "ignored_pred_indices": [],
                "pred_scope": "annotated",
                "iou_thr": 0.5,
            }
        },
    )

    row = _read_jsonl(out_path)[0]
    assert row["matching"]["match_source"] == "detection_eval"
    assert row["matching"]["match_policy"] == "f1ish_primary"
    assert row["matching"]["matched_pairs"] == [[0, 1]]
    assert row["matching"]["fn_gt_indices"] == [0]
    assert row["matching"]["fp_pred_indices"] == [1]
    assert row["matching"]["pred_scope"] == "annotated"


def test_materialize_vis_resource_preserves_explicit_source_indices_for_matching(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "image": "img.png",
                "width": 100,
                "height": 100,
                "coord_mode": "pixel",
                "gt": [
                    {"index": 10, "bbox_2d": [40, 40, 60, 60], "desc": "zebra"},
                    {"index": 20, "bbox_2d": [10, 10, 20, 20], "desc": "apple"},
                ],
                "pred": [
                    {"index": 7, "bbox_2d": [10, 10, 20, 20], "desc": "apple"},
                ],
                "matching": {
                    "match_source": "precomputed",
                    "match_policy": "unit_test",
                    "pred_index_domain": "source_pred_index",
                    "gt_index_domain": "source_gt_index",
                    "matched_pairs": [[7, 20]],
                    "fn_gt_indices": [10],
                    "fp_pred_indices": [],
                },
            }
        ],
    )

    row = _read_jsonl(materialize_gt_vs_pred_vis_resource(source_path))[0]
    assert [obj["index"] for obj in row["gt"]] == [0, 1]
    assert [obj["desc"] for obj in row["gt"]] == ["apple", "zebra"]
    assert [obj["index"] for obj in row["pred"]] == [7]
    assert row["matching"]["matched_pairs"] == [[7, 0]]
    assert row["matching"]["fn_gt_indices"] == [1]
    assert row["matching"]["fp_pred_indices"] == []


def test_materialize_eval_vis_resource_uses_matches_and_per_image_artifacts(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "gt_vs_pred.jsonl"
    matches_path = tmp_path / "eval" / "matches.jsonl"
    per_image_path = tmp_path / "eval" / "per_image.json"
    _write_jsonl(
        source_path,
        [
            {
                "line_idx": 12,
                "image": "img_eval.png",
                "width": 100,
                "height": 100,
                "coord_mode": "pixel",
                "gt": [
                    {"index": 30, "bbox_2d": [40, 40, 60, 60], "desc": "zebra"},
                    {"index": 20, "bbox_2d": [10, 10, 20, 20], "desc": "apple"},
                ],
                "pred": [
                    {"index": 5, "bbox_2d": [10, 10, 20, 20], "desc": "apple"},
                    {"index": 9, "bbox_2d": [80, 80, 90, 90], "desc": "noise"},
                ],
            }
        ],
    )
    _write_jsonl(
        matches_path,
        [
            {
                "image_id": 12,
                "matches": [{"pred_idx": 5, "gt_idx": 20}],
                "unmatched_gt_indices": [30],
                "unmatched_pred_indices": [9],
                "ignored_pred_indices": [],
                "pred_scope": "annotated",
                "iou_thr": 0.5,
            }
        ],
    )
    per_image_path.parent.mkdir(parents=True, exist_ok=True)
    per_image_path.write_text(
        json.dumps(
            [
                {
                    "image_id": 12,
                    "file_name": "img_eval.png",
                    "gt_count": 2,
                    "pred_count": 2,
                    "invalid_gt": False,
                    "invalid_pred": [],
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    row = _read_jsonl(
        materialize_eval_gt_vs_pred_vis_resource(
            source_path,
            matches_jsonl=matches_path,
            per_image_json=per_image_path,
        )
    )[0]
    assert row["record_idx"] == 12
    assert row["image_id"] == 12
    assert row["file_name"] == "img_eval.png"
    assert row["matching"]["matched_pairs"] == [[5, 0]]
    assert row["matching"]["fn_gt_indices"] == [1]
    assert row["matching"]["fp_pred_indices"] == [9]
    assert row["provenance"]["matches_jsonl"].endswith("matches.jsonl")


def test_materialize_vis_resource_accepts_scored_and_norm1000_inputs(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "gt_vs_pred_scored.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "image": "img.png",
                "width": 1000,
                "height": 1000,
                "coord_mode": "pixel",
                "gt": [
                    {
                        "index": 1,
                        "geom_type": "bbox_2d",
                        "points_norm1000": [0, 0, 999, 999],
                        "desc": "gt-box",
                    }
                ],
                "pred": [
                    {
                        "index": 4,
                        "type": "bbox_2d",
                        "points": [
                            "<|coord_0|>",
                            "<|coord_0|>",
                            "<|coord_999|>",
                            "<|coord_999|>",
                        ],
                        "desc": "pred-box",
                        "score": 0.97,
                    }
                ],
            }
        ],
    )

    row = _read_jsonl(materialize_gt_vs_pred_vis_resource(source_path))[0]
    assert row["gt"][0]["bbox_2d"] == [0, 0, 999, 999]
    assert row["pred"][0]["bbox_2d"] == [0, 0, 999, 999]
    assert row["matching"]["matched_pairs"] == [[4, 0]]


def test_materialize_vis_resource_handles_duplicate_gt_boxes_stably(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "image": "img.png",
                "width": 100,
                "height": 100,
                "coord_mode": "pixel",
                "gt": [
                    {"index": 9, "bbox_2d": [10, 10, 20, 20], "desc": "cat"},
                    {"index": 4, "bbox_2d": [10, 10, 20, 20], "desc": "cat"},
                    {"index": 7, "bbox_2d": [10, 10, 20, 20], "desc": "cat"},
                ],
                "pred": [],
                "matching": {
                    "match_source": "precomputed",
                    "match_policy": "unit_test",
                    "pred_index_domain": "source_pred_index",
                    "gt_index_domain": "source_gt_index",
                    "matched_pairs": [],
                    "fn_gt_indices": [9, 4, 7],
                    "fp_pred_indices": [],
                },
            }
        ],
    )

    row = _read_jsonl(materialize_gt_vs_pred_vis_resource(source_path))[0]
    assert [obj["desc"] for obj in row["gt"]] == ["cat", "cat", "cat"]
    assert [obj["index"] for obj in row["gt"]] == [0, 1, 2]
    assert row["matching"]["fn_gt_indices"] == [0, 1, 2]


def test_render_gt_vs_pred_review_rejects_missing_matching(tmp_path: Path) -> None:
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    resource_path = tmp_path / "vis_resources" / "gt_vs_pred.jsonl"
    _write_jsonl(
        resource_path,
        [
            {
                "schema_version": 1,
                "source_kind": "manual",
                "record_idx": 0,
                "image": "img.png",
                "width": 64,
                "height": 64,
                "coord_mode": "pixel",
                "gt": [{"index": 0, "desc": "a", "bbox_2d": [1, 1, 10, 10]}],
                "pred": [{"index": 0, "desc": "a", "bbox_2d": [2, 2, 11, 11]}],
            }
        ],
    )

    with pytest.raises(ValueError, match="missing canonical `matching`"):
        render_gt_vs_pred_review(
            resource_path,
            out_dir=tmp_path / "out",
            root_image_dir=tmp_path,
        )


def test_render_vis_from_jsonl_materializes_sidecar_and_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(tmp_path / "img.png")
    source_path = tmp_path / "gt_vs_pred.jsonl"
    _write_jsonl(
        source_path,
        [
            {
                "image": "img.png",
                "width": 64,
                "height": 64,
                "mode": "text",
                "coord_mode": "pixel",
                "gt": [_bbox("a", [1, 1, 10, 10])],
                "pred": [_bbox("a", [1, 1, 10, 10])],
                "errors": [],
            }
        ],
    )

    monkeypatch.setenv("ROOT_IMAGE_DIR", str(tmp_path))

    out_dir = tmp_path / "vis"
    render_vis_from_jsonl(source_path, out_dir=out_dir, limit=1)

    assert (out_dir / "vis_0000.png").exists()
    sidecar_path = tmp_path / "vis_resources" / "gt_vs_pred.jsonl"
    assert sidecar_path.exists()
    row = _read_jsonl(sidecar_path)[0]
    assert row["schema_version"] == 1
    assert row["matching"]["matched_pairs"] == [[0, 0]]


def test_render_gt_vs_pred_review_is_deterministic_for_crowded_labels(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (96, 96), color=(128, 128, 128)).save(tmp_path / "img.png")
    resource_path = tmp_path / "vis_resources" / "gt_vs_pred.jsonl"
    _write_jsonl(
        resource_path,
        [
            {
                "schema_version": 1,
                "source_kind": "manual",
                "record_idx": 0,
                "image": "img.png",
                "width": 96,
                "height": 96,
                "coord_mode": "pixel",
                "gt": [
                    {"index": 0, "desc": "cat", "bbox_2d": [8, 8, 40, 40]},
                    {"index": 1, "desc": "dog", "bbox_2d": [12, 12, 44, 44]},
                ],
                "pred": [
                    {"index": 0, "desc": "cat", "bbox_2d": [8, 8, 40, 40]},
                    {"index": 1, "desc": "wolf", "bbox_2d": [12, 12, 44, 44]},
                    {"index": 2, "desc": "bird", "bbox_2d": [16, 16, 48, 48]},
                ],
                "matching": {
                    "match_source": "manual",
                    "match_policy": "unit_test",
                    "pred_index_domain": "canonical_pred_index",
                    "gt_index_domain": "canonical_gt_index",
                    "matched_pairs": [[0, 0]],
                    "fn_gt_indices": [1],
                    "fp_pred_indices": [1, 2],
                },
            }
        ],
    )

    out_a = tmp_path / "render_a"
    out_b = tmp_path / "render_b"
    render_gt_vs_pred_review(resource_path, out_dir=out_a, root_image_dir=tmp_path)
    render_gt_vs_pred_review(resource_path, out_dir=out_b, root_image_dir=tmp_path)
    assert (out_a / "vis_0000.png").read_bytes() == (out_b / "vis_0000.png").read_bytes()


def test_compose_comparison_scenes_from_member_jsonls_preserves_member_pred_order(
    tmp_path: Path,
) -> None:
    hf_jsonl = tmp_path / "hf" / "pred_hf.jsonl"
    vllm_jsonl = tmp_path / "vllm" / "pred_vllm.jsonl"
    _write_jsonl(
        hf_jsonl,
        [
            {
                "line_idx": 9,
                "image": "img.png",
                "width": 64,
                "height": 64,
                "coord_mode": "pixel",
                "gt": [
                    {"index": 2, "bbox_2d": [20, 20, 30, 30], "desc": "zebra"},
                    {"index": 1, "bbox_2d": [1, 1, 10, 10], "desc": "apple"},
                ],
                "pred": [
                    {"index": 4, "bbox_2d": [20, 20, 30, 30], "desc": "zebra"},
                    {"index": 9, "bbox_2d": [1, 1, 10, 10], "desc": "apple"},
                ],
            }
        ],
    )
    _write_jsonl(
        vllm_jsonl,
        [
            {
                "line_idx": 9,
                "image": "img.png",
                "width": 64,
                "height": 64,
                "coord_mode": "pixel",
                "gt": [
                    {"index": 1, "bbox_2d": [1, 1, 10, 10], "desc": "apple"},
                    {"index": 2, "bbox_2d": [20, 20, 30, 30], "desc": "zebra"},
                ],
                "pred": [
                    {"index": 3, "bbox_2d": [1, 1, 10, 10], "desc": "apple"},
                ],
            }
        ],
    )

    scenes = compose_comparison_scenes_from_jsonls({"hf": hf_jsonl, "vllm": vllm_jsonl})
    assert len(scenes) == 1
    scene = scenes[0]
    assert scene["record_idx"] == 9
    assert [obj["desc"] for obj in scene["gt"]] == ["apple", "zebra"]
    member_map = {member["label"]: member["record"] for member in scene["members"]}
    assert [obj["index"] for obj in member_map["hf"]["pred"]] == [4, 9]
    assert [obj["index"] for obj in member_map["vllm"]["pred"]] == [3]


def test_compose_comparison_scenes_fails_fast_on_gt_mismatch(tmp_path: Path) -> None:
    hf_jsonl = tmp_path / "hf" / "pred_hf.jsonl"
    vllm_jsonl = tmp_path / "vllm" / "pred_vllm.jsonl"
    _write_jsonl(
        hf_jsonl,
        [
            {
                "line_idx": 3,
                "image": "img.png",
                "width": 64,
                "height": 64,
                "coord_mode": "pixel",
                "gt": [{"bbox_2d": [1, 1, 10, 10], "desc": "apple"}],
                "pred": [],
            }
        ],
    )
    _write_jsonl(
        vllm_jsonl,
        [
            {
                "line_idx": 3,
                "image": "img.png",
                "width": 64,
                "height": 64,
                "coord_mode": "pixel",
                "gt": [{"bbox_2d": [1, 1, 10, 10], "desc": "pear"}],
                "pred": [],
            }
        ],
    )

    with pytest.raises(ValueError, match="canonical GT mismatch"):
        compose_comparison_scenes_from_jsonls({"hf": hf_jsonl, "vllm": vllm_jsonl})
