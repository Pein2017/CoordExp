"""Focused tests for the sorted crossing owner-row geometry renderer.

Rather than driving the (concurrently evolving) real crossing/census pipeline,
these tests build a minimal, hand-constructed analysis directory that matches
exactly the ``geometry-visual-plan.json`` / ``geometry-analysis-receipt.json``
contract the renderer consumes.  Schema versions, file names, arms and the role
palette are read live from the analyzer module so the fixture tracks it, but no
analyzer geometry is re-derived here.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import analyze_sorted_crossing_owner_row_geometry as analyzer  # noqa: E402
from scripts.research import visualize_sorted_crossing_owner_row_geometry as viz  # noqa: E402

IMAGE_WIDTH = 240
IMAGE_HEIGHT = 160
IMAGE_FILES = {"11": "images/val2017/000000000011.png", "22": "images/val2017/000000000022.png"}

PRIMARY_ARM = analyzer.PRIMARY_ARM
SENSITIVITY_ARM = analyzer.SENSITIVITY_ARM

#: ``(arm, gt_owner_id, image_id, material_negative)`` for both scatter arms.
PRIMARY_ROWS = (
    (PRIMARY_ARM, "gt:11:1", "11", True),
    (PRIMARY_ARM, "gt:11:2", "11", False),
    (PRIMARY_ARM, "gt:22:3", "22", False),
)
SENSITIVITY_ROWS = (
    (SENSITIVITY_ARM, "gt:11:1", "11", False),
    (SENSITIVITY_ARM, "gt:11:2", "11", False),
    (SENSITIVITY_ARM, "gt:22:3", "22", True),
)
GREEDY_PAIRS = (
    ("gt:11:5", "gt:11:9", "11"),
    ("gt:22:6", "gt:22:8", "22"),
)

C_BOX = [100, 100, 300, 300]
R_BOX = [200, 200, 400, 400]
CR_WINDOW = {"x1": 60, "y1": 60, "x2": 440, "y2": 440}
TARGET_BOX = [500, 500, 700, 700]
DISPLACER_BOX = [520, 520, 720, 720]
PAIR_WINDOW = {"x1": 460, "y1": 460, "x2": 760, "y2": 760}


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


def write_media(media_root: Path, *, file_names: dict[str, str] | None = None) -> Path:
    """Write one deterministic synthetic image per plan image_id."""

    for index, (image_id, name) in enumerate(sorted((file_names or IMAGE_FILES).items())):
        path = media_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (20 + index * 30, 60, 90))
        for x in range(0, IMAGE_WIDTH, 8):
            for y in range(0, IMAGE_HEIGHT, 8):
                image.putpixel((x, y), (200, 200, 200))
        image.save(path)
        assert image_id
    return media_root


def _scatter_panel(arm: str, measure: str, rows: tuple, *, arm_role: str) -> dict:
    label = "downstream_row_e" if arm == PRIMARY_ARM else "downstream_row_f"
    return {
        "panel_id": f"scatter:{arm}:{measure}",
        "output_file": f"scatter-{arm}-{measure}.png",
        "arm": arm,
        "arm_role": arm_role,
        "x_axis": {"field": measure, "label": measure.replace("_", " ")},
        "y_axis": {
            "field": "relative_coordinate_delta",
            "label": "relative coordinate delta (nat)",
        },
        "reference_lines": [
            {
                "axis": "y",
                "value": analyzer.MATERIAL_NEGATIVE_MAX_NATS,
                "label": "material_negative cutoff",
            }
        ],
        "points": [
            {
                "point_id": f"{arm}:{owner_id}",
                "image_id": image_id,
                "gt_owner_id": owner_id,
                "x": 0.1 * (index + 1),
                "y": -1.5 if material else -0.25 * (index + 1),
                "geometric_label": analyzer.LABEL_SEPARATED,
                "description_axis": "same_description",
                "material_negative": material,
                "color": analyzer.ROLE_PALETTE[label],
            }
            for index, (_arm, owner_id, image_id, material) in enumerate(rows)
        ],
    }


def _material_negative_crop(arm: str, owner_id: str, image_id: str, media_root: str) -> dict:
    label = "E" if arm == PRIMARY_ARM else "F"
    role = "downstream_row_e" if arm == PRIMARY_ARM else "downstream_row_f"
    return {
        "panel_id": f"crop:{arm}:{owner_id}",
        "output_file": f"crop-{arm}-{image_id}-{owner_id.replace(':', '_')}.png",
        "panel_kind": "material_negative_case",
        "arm": arm,
        "arm_role": "primary" if arm == PRIMARY_ARM else "sensitivity",
        "image": _image_block(image_id, media_root),
        "crop_window_norm1000_xyxy": dict(CR_WINDOW),
        "boxes": [
            {
                "box_id": f"C:{owner_id}",
                "role": "inserted_owner_c",
                "legend": f"C {owner_id} (person)",
                "norm1000_xyxy": list(C_BOX),
                "color": analyzer.ROLE_PALETTE["inserted_owner_c"],
            },
            {
                "box_id": f"{label}:{owner_id}",
                "role": role,
                "legend": f"{label} row 3 (person, matched)",
                "norm1000_xyxy": list(R_BOX),
                "color": analyzer.ROLE_PALETTE[role],
            },
        ],
        "annotations": {
            "iou": 0.14,
            "center_distance_normalized": 0.1,
            "relative_coordinate_delta": -1.5,
            "geometric_label": analyzer.LABEL_SEPARATED,
            "description_axis": "same_description",
            "sorted_key_rank_gap": 2,
        },
    }


def _greedy_crop(target_id: str, displacer_id: str, image_id: str, media_root: str) -> dict:
    return {
        "panel_id": f"crop:greedy_pair:{target_id}",
        "output_file": f"crop-greedy-pair-{image_id}-{target_id.replace(':', '_')}.png",
        "panel_kind": "greedy_displacement_pair",
        "arm": "greedy_displacement_pair",
        "arm_role": "descriptive_compatibility_check",
        "image": _image_block(image_id, media_root),
        "crop_window_norm1000_xyxy": dict(PAIR_WINDOW),
        "boxes": [
            {
                "box_id": f"target:{target_id}",
                "role": "displacement_target",
                "legend": f"target {target_id} (person)",
                "norm1000_xyxy": list(TARGET_BOX),
                "color": analyzer.ROLE_PALETTE["displacement_target"],
            },
            {
                "box_id": f"displacer:{displacer_id}",
                "role": "displacer",
                "legend": f"displacer {displacer_id} (person)",
                "norm1000_xyxy": list(DISPLACER_BOX),
                "color": analyzer.ROLE_PALETTE["displacer"],
            },
        ],
        "annotations": {
            "iou": 0.68,
            "center_distance_normalized": 0.02,
            "geometric_label": analyzer.LABEL_HIGH_OVERLAP,
            "sorted_key_rank_gap": 1,
            "displacer_native_true_positive": True,
        },
    }


def _image_block(image_id: str, media_root: str, *, file_name: str | None = None) -> dict:
    return {
        "image_id": image_id,
        "media_root": media_root,
        "file_name": file_name or IMAGE_FILES[image_id],
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "executed_media_sha256": "0" * 64,
        "media_identity_note": "executed media tensor, not the file on disk",
    }


def build_plan(media_root: str) -> dict:
    scatter_panels = [
        _scatter_panel(PRIMARY_ARM, measure, PRIMARY_ROWS, arm_role="primary")
        for measure in viz.SCATTER_MEASURES
    ] + [
        _scatter_panel(SENSITIVITY_ARM, measure, SENSITIVITY_ROWS, arm_role="sensitivity")
        for measure in viz.SCATTER_MEASURES
    ]
    crop_panels = [
        _material_negative_crop(PRIMARY_ARM, "gt:11:1", "11", media_root),
        _material_negative_crop(SENSITIVITY_ARM, "gt:22:3", "22", media_root),
    ] + [
        _greedy_crop(target, displacer, image_id, media_root)
        for target, displacer, image_id in GREEDY_PAIRS
    ]
    panels = [*scatter_panels, *crop_panels]
    return {
        "schema_version": analyzer.VISUAL_PLAN_SCHEMA_VERSION,
        "unit_id": analyzer.UNIT_ID,
        "plan_role": "machine-readable render plan and manifest",
        "palette": dict(analyzer.ROLE_PALETTE),
        "render_projection": "norm-1000 bins times the sealed image extent, display only",
        "material_negative_case_count": 2,
        "greedy_pair_panel_count": len(GREEDY_PAIRS),
        "panel_count": len(panels),
        "expected_output_files": sorted(panel["output_file"] for panel in panels),
        "scatter_panels": scatter_panels,
        "crop_panels": crop_panels,
    }


def build_denominators() -> dict:
    return {
        "crossing_owner_count": len(PRIMARY_ROWS),
        "primary_ce_row_count": len(PRIMARY_ROWS),
        "sensitivity_cf_row_count": len(SENSITIVITY_ROWS),
        "greedy_displacement_pair_count": len(GREEDY_PAIRS),
        "benign_control_count": 2,
        "image_count": 2,
    }


def write_analysis_dir(
    root: Path,
    *,
    plan: dict | None = None,
    denominators: dict | None = None,
    media_root: str | None = None,
) -> Path:
    """Write a minimal but contract-exact analysis directory and seal it."""

    analysis_dir = root / "geometry-analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plan = plan if plan is not None else build_plan(media_root or str(root / "media"))
    files = {
        analyzer.OWNER_ROWS_NAME: b'{"row": 1}\n',
        analyzer.PAIR_ROWS_NAME: b'{"pair": 1}\n',
        analyzer.SUMMARY_NAME: b'{"summary": true}\n',
        analyzer.REPORT_MD_NAME: b"# report\n",
        analyzer.VISUAL_PLAN_NAME: (
            json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
    }
    receipt = {
        "schema_version": analyzer.RECEIPT_SCHEMA_VERSION,
        "unit_id": analyzer.UNIT_ID,
        "source_unit_id": analyzer.SOURCE_UNIT_ID,
        "denominators": denominators if denominators is not None else build_denominators(),
        "output_file_digests": {
            name: {
                "path": name,
                "byte_size": len(payload),
                "sha256": analyzer.sha256_bytes(payload),
            }
            for name, payload in files.items()
        },
    }
    receipt["receipt_content_sha256"] = analyzer.sha256_json(receipt)
    for name, payload in files.items():
        (analysis_dir / name).write_bytes(payload)
    (analysis_dir / analyzer.RECEIPT_NAME).write_bytes(
        analyzer.canonical_json_bytes(receipt) + b"\n"
    )
    return analysis_dir


def reseal(analysis_dir: Path, plan: dict) -> None:
    """Rewrite the plan *and* the receipt digest that seals it."""

    payload = (json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    (analysis_dir / analyzer.VISUAL_PLAN_NAME).write_bytes(payload)
    receipt = json.loads((analysis_dir / analyzer.RECEIPT_NAME).read_text(encoding="utf-8"))
    receipt["output_file_digests"][analyzer.VISUAL_PLAN_NAME] = {
        "path": analyzer.VISUAL_PLAN_NAME,
        "byte_size": len(payload),
        "sha256": analyzer.sha256_bytes(payload),
    }
    receipt.pop("receipt_content_sha256", None)
    receipt["receipt_content_sha256"] = analyzer.sha256_json(receipt)
    (analysis_dir / analyzer.RECEIPT_NAME).write_bytes(
        analyzer.canonical_json_bytes(receipt) + b"\n"
    )


def read_plan(analysis_dir: Path) -> dict:
    return json.loads((analysis_dir / analyzer.VISUAL_PLAN_NAME).read_text(encoding="utf-8"))


@pytest.fixture()
def analysis_dir(tmp_path: Path) -> Path:
    write_media(tmp_path / "media")
    return write_analysis_dir(tmp_path)


# ---------------------------------------------------------------------------
# Happy path: exact panel coverage
# ---------------------------------------------------------------------------


def test_render_publishes_every_scatter_and_crop_panel(analysis_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "viz"
    result = viz.run_visualization(analysis_dir=analysis_dir, output_dir=output_dir)

    assert result["published"]["published"] is True
    assert result["panel_count"] == 8
    assert result["rendered_panel_count"] == 8

    plan = read_plan(analysis_dir)
    rendered = sorted(path.name for path in output_dir.glob("*.png"))
    assert rendered == sorted(plan["expected_output_files"])
    assert len(rendered) == 4 + 4

    manifest = json.loads((output_dir / viz.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in manifest["panels"]] == [
        *(panel["panel_id"] for panel in plan["scatter_panels"]),
        *(panel["panel_id"] for panel in plan["crop_panels"]),
    ]
    assert manifest["visual_plan_sha256"] == analyzer.sha256_bytes(
        (analysis_dir / analyzer.VISUAL_PLAN_NAME).read_bytes()
    )
    assert manifest["analysis_receipt_content_sha256"] == json.loads(
        (analysis_dir / analyzer.RECEIPT_NAME).read_text(encoding="utf-8")
    )["receipt_content_sha256"]

    # Exact point and box IDs, and an output relative path + sha256 for every panel.
    by_id = {panel["panel_id"]: panel for panel in manifest["panels"]}
    assert by_id[f"scatter:{PRIMARY_ARM}:iou"]["point_ids"] == [
        f"{PRIMARY_ARM}:{owner}" for _arm, owner, _image, _material in PRIMARY_ROWS
    ]
    assert by_id[f"crop:{PRIMARY_ARM}:gt:11:1"]["box_ids"] == ["C:gt:11:1", "E:gt:11:1"]
    assert by_id["crop:greedy_pair:gt:11:5"]["box_ids"] == [
        "target:gt:11:5",
        "displacer:gt:11:9",
    ]
    for panel in manifest["panels"]:
        path = output_dir / panel["output_relative_path"]
        assert path.is_file()
        assert analyzer.sha256_bytes(path.read_bytes()) == panel["sha256"]
        assert panel["byte_size"] == path.stat().st_size

    # The self-seal covers every published byte except the manifest itself.
    digests = manifest["output_file_digests"]
    assert set(digests) == {
        *plan["expected_output_files"],
        viz.PANEL_SPECS_NAME,
        viz.INDEX_NAME,
    }
    assert (output_dir / viz.INDEX_NAME).read_text(encoding="utf-8").startswith("# Sorted")


def test_material_negative_cases_and_greedy_pairs_are_both_covered(
    analysis_dir: Path, tmp_path: Path
) -> None:
    viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")
    manifest = json.loads((tmp_path / "viz" / viz.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["coverage"]["material_negative_case_count"] == 2
    assert manifest["coverage"]["greedy_pair_panel_count"] == len(GREEDY_PAIRS)

    specs = [
        json.loads(line)
        for line in (tmp_path / "viz" / viz.PANEL_SPECS_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    material = [spec for spec in specs if spec.get("crop_kind") == "material_negative_case"]
    # One crop panel for exactly the material-negative rows of both arms.
    assert {spec["panel_id"] for spec in material} == {
        f"crop:{PRIMARY_ARM}:gt:11:1",
        f"crop:{SENSITIVITY_ARM}:gt:22:3",
    }


def test_missing_crop_panel_for_a_material_negative_row_fails_closed(
    analysis_dir: Path, tmp_path: Path
) -> None:
    plan = read_plan(analysis_dir)
    plan["crop_panels"] = [
        panel for panel in plan["crop_panels"] if panel["panel_id"] != f"crop:{PRIMARY_ARM}:gt:11:1"
    ]
    plan["material_negative_case_count"] = 1
    plan["panel_count"] = len(plan["scatter_panels"]) + len(plan["crop_panels"])
    plan["expected_output_files"] = sorted(
        panel["output_file"] for panel in [*plan["scatter_panels"], *plan["crop_panels"]]
    )
    reseal(analysis_dir, plan)
    with pytest.raises(
        viz.VisualContractError, match="material-negative rows with no crop panel"
    ):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_missing_scatter_arm_measure_panel_fails_closed(
    analysis_dir: Path, tmp_path: Path
) -> None:
    plan = read_plan(analysis_dir)
    plan["scatter_panels"] = [
        panel
        for panel in plan["scatter_panels"]
        if panel["panel_id"] != f"scatter:{SENSITIVITY_ARM}:iou"
    ]
    plan["panel_count"] = len(plan["scatter_panels"]) + len(plan["crop_panels"])
    plan["expected_output_files"] = sorted(
        panel["output_file"] for panel in [*plan["scatter_panels"], *plan["crop_panels"]]
    )
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="every arm x measure pair"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_scatter_point_count_must_equal_the_sealed_owner_denominator(
    analysis_dir: Path, tmp_path: Path
) -> None:
    plan = read_plan(analysis_dir)
    plan["scatter_panels"][0]["points"] = plan["scatter_panels"][0]["points"][:-1]
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="owner rows the receipt sealed"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


# ---------------------------------------------------------------------------
# Projection and role encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "extent", "expected"),
    [
        (0, 1024, 0.0),
        (500, 1024, 512.0),
        (1000, 1024, 1024.0),
        (250, 240, 60.0),
        (1000, 160, 160.0),
    ],
)
def test_norm1000_projects_linearly_into_the_sealed_pixel_extent(
    value: int, extent: int, expected: float
) -> None:
    assert viz.project_norm1000_to_pixel(value, extent) == pytest.approx(expected)


def test_projection_rejects_a_nonpositive_extent() -> None:
    with pytest.raises(viz.VisualContractError, match="nonpositive image extent"):
        viz.project_norm1000_to_pixel(500, 0)


def test_crop_spec_projects_the_window_and_keeps_stable_role_ids(analysis_dir: Path) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    specs = viz.build_panel_specs(artifacts, image_root=None)
    crop = next(spec for spec in specs if spec["panel_id"] == f"crop:{PRIMARY_ARM}:gt:11:1")

    assert crop["crop_window_pixel_xyxy"] == [
        int(CR_WINDOW["x1"] * IMAGE_WIDTH / 1000),
        int(CR_WINDOW["y1"] * IMAGE_HEIGHT / 1000),
        -(-CR_WINDOW["x2"] * IMAGE_WIDTH // 1000),
        -(-CR_WINDOW["y2"] * IMAGE_HEIGHT // 1000),
    ]
    assert crop["box_ids"] == ["C:gt:11:1", "E:gt:11:1"]
    assert [box["role"] for box in crop["boxes"]] == ["inserted_owner_c", "downstream_row_e"]
    assert [box["color"] for box in crop["boxes"]] == [
        analyzer.ROLE_PALETTE["inserted_owner_c"],
        analyzer.ROLE_PALETTE["downstream_row_e"],
    ]
    assert crop["boxes"][0]["legend"].startswith("C gt:11:1")
    # Supplied geometry/delta annotations are carried, never recomputed.
    assert crop["annotations"]["relative_coordinate_delta"] == -1.5
    assert {"key": "iou", "text": "0.1400"} in crop["annotation_lines"]


def test_rendered_crop_shows_both_role_colors(analysis_dir: Path, tmp_path: Path) -> None:
    viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")
    for panel_id, roles in (
        (f"crop:{PRIMARY_ARM}:gt:11:1", ("inserted_owner_c", "downstream_row_e")),
        ("crop:greedy_pair:gt:11:5", ("displacement_target", "displacer")),
    ):
        manifest = json.loads((tmp_path / "viz" / viz.MANIFEST_NAME).read_text(encoding="utf-8"))
        panel = next(item for item in manifest["panels"] if item["panel_id"] == panel_id)
        with Image.open(tmp_path / "viz" / panel["output_relative_path"]) as handle:
            colors = {color for _count, color in handle.convert("RGB").getcolors(1 << 20)}
        for role in roles:
            expected = viz._rgb(analyzer.ROLE_PALETTE[role], "role")
            assert expected in colors, f"{panel_id} does not draw role {role}"


def test_two_boxes_drawn_in_the_same_color_fail_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    plan["palette"]["downstream_row_e"] = plan["palette"]["inserted_owner_c"]
    for panel in plan["crop_panels"]:
        for box in panel["boxes"]:
            if box["role"] == "downstream_row_e":
                box["color"] = plan["palette"]["inserted_owner_c"]
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="indistinguishable"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_box_color_must_match_the_plan_palette_for_its_role(
    analysis_dir: Path, tmp_path: Path
) -> None:
    plan = read_plan(analysis_dir)
    plan["crop_panels"][0]["boxes"][1]["color"] = "#00ff00"
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="not the plan palette color"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


# ---------------------------------------------------------------------------
# Media resolution
# ---------------------------------------------------------------------------


def test_declared_root_join_is_the_resolution_rule_for_the_owning_artifact(
    analysis_dir: Path,
) -> None:
    artifacts = viz.load_artifacts(analysis_dir)
    specs = viz.build_panel_specs(artifacts, image_root=None)
    crop = next(spec for spec in specs if spec["panel_kind"] == "crop")
    assert crop["image"]["media_resolution_rule"] == viz.RULE_DECLARED_ROOT_JOIN
    assert crop["image"]["media_root_source"] == "visual_plan_media_root"
    assert crop["image"]["resolved_path"].endswith(IMAGE_FILES[crop["image"]["image_id"]])


def test_media_root_that_already_ends_with_the_file_name_prefix_resolves_once(
    tmp_path: Path,
) -> None:
    """The superseded plan shape: media_root already carries ``images/val2017``."""

    media_root = tmp_path / "media"
    write_media(media_root)
    overlapping_root = str(media_root / "images" / "val2017")
    analysis_dir = write_analysis_dir(tmp_path, media_root=overlapping_root)

    artifacts = viz.load_artifacts(analysis_dir)
    specs = viz.build_panel_specs(artifacts, image_root=None)
    crop = next(spec for spec in specs if spec["panel_kind"] == "crop")
    assert crop["image"]["media_resolution_rule"] == viz.RULE_DECLARED_ROOT_OVERLAP_STRIPPED
    assert Path(crop["image"]["resolved_path"]) == media_root / IMAGE_FILES[
        crop["image"]["image_id"]
    ]


def test_two_resolvable_media_roots_fail_closed_as_ambiguous(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    write_media(media_root)
    overlapping_root = media_root / "images" / "val2017"
    # Make the naive duplicated join exist too, so neither candidate is provable.
    write_media(overlapping_root)
    analysis_dir = write_analysis_dir(tmp_path, media_root=str(overlapping_root))
    with pytest.raises(viz.VisualContractError, match="ambiguous"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_missing_media_file_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "media" / IMAGE_FILES["11"]).unlink()
    with pytest.raises(viz.VisualContractError, match="media file for image 11 is missing"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_media_file_with_wrong_pixel_dimensions_fails_closed(
    analysis_dir: Path, tmp_path: Path
) -> None:
    Image.new("RGB", (IMAGE_WIDTH + 3, IMAGE_HEIGHT), (10, 10, 10)).save(
        tmp_path / "media" / IMAGE_FILES["11"]
    )
    with pytest.raises(viz.VisualContractError, match="not the sealed"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_file_name_escaping_its_root_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    plan["crop_panels"][0]["image"]["file_name"] = "../../etc/passwd"
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="escapes its media root"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_image_root_override_replaces_the_declared_root(
    analysis_dir: Path, tmp_path: Path
) -> None:
    other_root = tmp_path / "elsewhere"
    write_media(other_root)
    artifacts = viz.load_artifacts(analysis_dir)
    specs = viz.build_panel_specs(artifacts, image_root=other_root)
    crop = next(spec for spec in specs if spec["panel_kind"] == "crop")
    assert crop["image"]["media_root_source"] == "cli_image_root_override"
    assert Path(crop["image"]["resolved_path"]).is_relative_to(other_root)


# ---------------------------------------------------------------------------
# Digest, schema and structural failures
# ---------------------------------------------------------------------------


def test_visual_plan_digest_drift_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    plan["panel_count"] = 99
    # Rewrite the plan *without* resealing the receipt.
    (analysis_dir / analyzer.VISUAL_PLAN_NAME).write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(viz.VisualContractError, match="does not match the digest sealed"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_receipt_self_seal_tamper_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    receipt = json.loads((analysis_dir / analyzer.RECEIPT_NAME).read_text(encoding="utf-8"))
    receipt["denominators"]["greedy_displacement_pair_count"] = 99
    (analysis_dir / analyzer.RECEIPT_NAME).write_bytes(
        analyzer.canonical_json_bytes(receipt) + b"\n"
    )
    with pytest.raises(viz.VisualContractError, match="does not reconstruct its own digest"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_visual_plan_schema_drift_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    plan["schema_version"] = "sorted-crossing-owner-row-geometry-visual-plan.v2"
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="schema"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_foreign_unit_id_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    plan["unit_id"] = "2026-01-01-some-other-unit"
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="belongs to another unit"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_unsealed_file_in_the_analysis_directory_fails_closed(
    analysis_dir: Path, tmp_path: Path
) -> None:
    (analysis_dir / "stray-notes.md").write_text("hand edit\n", encoding="utf-8")
    with pytest.raises(viz.VisualContractError, match="never sealed"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_duplicate_panel_id_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    duplicate = dict(plan["crop_panels"][0])
    duplicate["output_file"] = "crop-duplicate-id.png"
    plan["crop_panels"].append(duplicate)
    plan["material_negative_case_count"] += 1
    plan["panel_count"] += 1
    plan["expected_output_files"] = sorted(
        [*plan["expected_output_files"], duplicate["output_file"]]
    )
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="panel_id contains the duplicate ID"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_duplicate_output_file_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    plan["crop_panels"][1]["output_file"] = plan["crop_panels"][0]["output_file"]
    plan["expected_output_files"] = sorted(
        panel["output_file"] for panel in [*plan["scatter_panels"], *plan["crop_panels"]]
    )
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="output_file contains the duplicate ID"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_duplicate_box_id_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    panel = plan["crop_panels"][0]
    panel["boxes"][1]["box_id"] = panel["boxes"][0]["box_id"]
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="box_id contains the duplicate ID"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_duplicate_point_id_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    points = plan["scatter_panels"][0]["points"]
    points[1]["point_id"] = points[0]["point_id"]
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="point_id contains the duplicate ID"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


@pytest.mark.parametrize(
    "box",
    [
        [300, 100, 300, 300],  # x2 == x1
        [100, 300, 300, 300],  # y2 == y1
        [-5, 100, 300, 300],  # outside the norm-1000 canvas
        [100, 100, 1200, 300],  # outside the norm-1000 canvas
    ],
)
def test_invalid_norm1000_box_fails_closed(
    analysis_dir: Path, tmp_path: Path, box: list[int]
) -> None:
    plan = read_plan(analysis_dir)
    plan["crop_panels"][0]["boxes"][0]["norm1000_xyxy"] = box
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="norm1000_xyxy|crop window"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_box_outside_its_own_crop_window_fails_closed(
    analysis_dir: Path, tmp_path: Path
) -> None:
    plan = read_plan(analysis_dir)
    plan["crop_panels"][0]["boxes"][0]["norm1000_xyxy"] = [10, 10, 50, 50]
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="outside the panel's own crop window"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_invalid_crop_window_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    plan["crop_panels"][0]["crop_window_norm1000_xyxy"] = {
        "x1": 400,
        "y1": 60,
        "x2": 400,
        "y2": 440,
    }
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="requires x2 > x1"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_nonfinite_scatter_coordinate_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    plan["scatter_panels"][0]["points"][0]["y"] = float("nan")
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="is not finite"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_unknown_crop_panel_kind_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    plan["crop_panels"][0]["panel_kind"] = "something_else"
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="unknown panel_kind"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


def test_mismatched_role_pair_fails_closed(analysis_dir: Path, tmp_path: Path) -> None:
    plan = read_plan(analysis_dir)
    panel = next(
        item for item in plan["crop_panels"] if item["panel_kind"] == "material_negative_case"
    )
    panel["boxes"][1]["role"] = "displacer"
    panel["boxes"][1]["color"] = analyzer.ROLE_PALETTE["displacer"]
    reseal(analysis_dir, plan)
    with pytest.raises(viz.VisualContractError, match="inserted owner C with its downstream row"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=tmp_path / "viz")


# ---------------------------------------------------------------------------
# Publish semantics
# ---------------------------------------------------------------------------


def test_identical_rerun_is_a_no_op_and_a_drift_fails_closed(
    analysis_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "viz"
    first = viz.run_visualization(analysis_dir=analysis_dir, output_dir=output_dir)
    assert first["published"]["published"] is True
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    second = viz.run_visualization(analysis_dir=analysis_dir, output_dir=output_dir)
    assert second["published"]["published"] is False
    assert second["published"]["publish_mode"] == "no_op_identical_rerun"
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before

    (output_dir / viz.INDEX_NAME).write_text("hand edited\n", encoding="utf-8")
    with pytest.raises(viz.VisualContractError, match="not a byte-identical rerun"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=output_dir)
    assert (output_dir / viz.INDEX_NAME).read_text(encoding="utf-8") == "hand edited\n"


def test_specs_only_publishes_validated_specs_without_any_image(
    analysis_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "viz-specs"
    result = viz.run_visualization(
        analysis_dir=analysis_dir, output_dir=output_dir, specs_only=True
    )
    assert result["specs_only"] is True
    assert result["panel_count"] == 8
    assert result["rendered_panel_count"] == 0
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        [viz.PANEL_SPECS_NAME, viz.MANIFEST_NAME, viz.INDEX_NAME]
    )
    manifest = json.loads((output_dir / viz.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["specs_only"] is True
    assert all(panel["rendered"] is False for panel in manifest["panels"])
    assert all(panel["sha256"] is None for panel in manifest["panels"])
    # Media is still resolved and verified, so specs-only proves the same preconditions.
    assert all(
        panel["media_file_sha256"] is not None
        for panel in manifest["panels"]
        if panel["panel_kind"] == "crop"
    )


def test_specs_only_is_deterministic(analysis_dir: Path, tmp_path: Path) -> None:
    first = tmp_path / "specs-a"
    second = tmp_path / "specs-b"
    viz.run_visualization(analysis_dir=analysis_dir, output_dir=first, specs_only=True)
    viz.run_visualization(analysis_dir=analysis_dir, output_dir=second, specs_only=True)
    assert (first / viz.PANEL_SPECS_NAME).read_bytes() == (
        second / viz.PANEL_SPECS_NAME
    ).read_bytes()

    rerun = viz.run_visualization(
        analysis_dir=analysis_dir, output_dir=first, specs_only=True
    )
    assert rerun["published"]["published"] is False


def test_specs_only_and_rendered_runs_cannot_share_one_directory(
    analysis_dir: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "viz"
    viz.run_visualization(analysis_dir=analysis_dir, output_dir=output_dir, specs_only=True)
    with pytest.raises(viz.VisualContractError, match="not a byte-identical rerun"):
        viz.run_visualization(analysis_dir=analysis_dir, output_dir=output_dir)


def test_main_reports_a_contract_violation_as_a_nonzero_exit(
    analysis_dir: Path, tmp_path: Path
) -> None:
    (analysis_dir / analyzer.VISUAL_PLAN_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        viz.main(
            [
                "--analysis-dir",
                str(analysis_dir),
                "--output-dir",
                str(tmp_path / "viz"),
                "--specs-only",
            ]
        )
    assert "visualization contract violated" in str(excinfo.value)
