"""Focused tests for the full-canvas visual-token-budget intervention preparation.

CPU-only and repo-data-free: the byte-exact reproduction proof runs against a
JPEG this test writes into ``tmp_path``, never against ``public_data`` or the
predecessor run root.  The twelve target dimensions are pinned as *constants*
derived from the frozen raw COCO sizes, so a change in the aligned-dimension
copy fails here rather than silently in a GPU capture.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as planner  # noqa: E402
from scripts.research import (  # noqa: E402
    prepare_sorted_full_canvas_token_budget_intervention as prepare,
)


# ---------------------------------------------------------------------------
# The twelve panel images
# ---------------------------------------------------------------------------

#: ``image_id -> (raw_width, raw_height, current_w, current_h, treatment_w, treatment_h)``
#: Raw sizes are the frozen COCO val2017 sizes of the human-refined-12 panel;
#: the current sizes are the sealed census image-registry canvases.
PANEL: dict[str, tuple[int, int, int, int, int, int]] = {
    "1584": (612, 612, 1024, 1024, 1440, 1440),
    "2685": (640, 555, 1024, 896, 1504, 1312),
    "4134": (640, 425, 1248, 832, 1728, 1152),
    "5001": (640, 480, 1152, 864, 1664, 1248),
    "6040": (640, 351, 1344, 736, 1920, 1056),
    "7511": (640, 480, 1152, 864, 1664, 1248),
    "10707": (640, 480, 1152, 864, 1664, 1248),
    "13348": (640, 427, 1248, 832, 1728, 1152),
    "13923": (640, 427, 1248, 832, 1728, 1152),
    "14038": (640, 427, 1248, 832, 1728, 1152),
    "14439": (640, 404, 1216, 768, 1728, 1088),
    "16228": (640, 440, 1216, 832, 1728, 1184),
}


def test_all_twelve_current_pool_dimensions_reproduce_from_raw_sizes():
    for image_id, (raw_w, raw_h, cur_w, cur_h, _, _) in PANEL.items():
        height, width = prepare.aligned_dimensions(
            height=raw_h, width=raw_w, max_pixels=prepare.BASELINE_MAX_PIXELS
        )
        assert (width, height) == (cur_w, cur_h), image_id


def test_all_twelve_treatment_dimensions_and_roughly_double_visual_tokens():
    ratios = []
    for image_id, (raw_w, raw_h, cur_w, cur_h, trt_w, trt_h) in PANEL.items():
        height, width = prepare.aligned_dimensions(
            height=raw_h, width=raw_w, max_pixels=prepare.TREATMENT_MAX_PIXELS
        )
        assert (width, height) == (trt_w, trt_h), image_id
        assert width * height <= prepare.TREATMENT_MAX_PIXELS, image_id
        assert width % prepare.IMAGE_FACTOR == 0 and height % prepare.IMAGE_FACTOR == 0

        current_tokens = prepare.merged_visual_token_count(width=cur_w, height=cur_h)
        treatment_tokens = prepare.merged_visual_token_count(width=width, height=height)
        assert current_tokens <= 1024, image_id
        assert 1024 < treatment_tokens <= 2048, image_id
        ratios.append(treatment_tokens / current_tokens)
        assert 1.9 <= ratios[-1] <= 2.2, (image_id, ratios[-1])

    pooled = sum(
        prepare.merged_visual_token_count(width=row[4], height=row[5]) for row in PANEL.values()
    ) / sum(
        prepare.merged_visual_token_count(width=row[2], height=row[3]) for row in PANEL.values()
    )
    assert 1.95 <= pooled <= 2.05


def test_treatment_keeps_aspect_ratio_within_the_grid_quantization():
    for image_id, (raw_w, raw_h, _, _, trt_w, trt_h) in PANEL.items():
        raw_aspect = raw_w / raw_h
        treatment_aspect = trt_w / trt_h
        assert abs(treatment_aspect / raw_aspect - 1.0) < 0.06, image_id


# ---------------------------------------------------------------------------
# Byte-exact reproduction of the current pool
# ---------------------------------------------------------------------------


def _write_raw_jpeg(path: Path, *, width: int, height: int) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 3) % 256)
    image.save(path, format="JPEG")
    return path


def test_current_pool_reproduction_is_byte_exact_and_fails_closed(tmp_path: Path):
    raw = _write_raw_jpeg(tmp_path / "raw.jpg", width=320, height=240)
    height, width = prepare.aligned_dimensions(
        height=240, width=320, max_pixels=prepare.BASELINE_MAX_PIXELS
    )
    payload = prepare.render_pool_jpeg_bytes(raw, width=width, height=height)
    identity = prepare.decoded_media_identity(payload)

    registry = {
        "image_width": width,
        "image_height": height,
        "executed_media_sha256": identity["executed_media_sha256"],
    }
    report = prepare.reproduce_current_pool(raw_path=raw, registry=registry)
    assert report["reproduced"] is True
    assert report["dimensions_match"] is True
    assert report["executed_media_sha256_match"] is True
    assert report["executed_media_sha256"] == identity["executed_media_sha256"]

    # A different digest for the same canvas is a reproduction failure, not a
    # rounding difference to be waved through.
    drifted = dict(registry, executed_media_sha256="0" * 64)
    assert prepare.reproduce_current_pool(raw_path=raw, registry=drifted)["reproduced"] is False

    resized = dict(registry, image_width=width + 32)
    assert prepare.reproduce_current_pool(raw_path=raw, registry=resized)["reproduced"] is False


def test_rendering_is_deterministic_and_pixel_digest_tracks_content(tmp_path: Path):
    raw = _write_raw_jpeg(tmp_path / "raw.jpg", width=320, height=240)
    first = prepare.render_pool_jpeg_bytes(raw, width=256, height=192)
    second = prepare.render_pool_jpeg_bytes(raw, width=256, height=192)
    assert first == second
    denser = prepare.render_pool_jpeg_bytes(raw, width=384, height=288)
    assert (
        prepare.decoded_media_identity(first)["executed_media_sha256"]
        != prepare.decoded_media_identity(denser)["executed_media_sha256"]
    )


# ---------------------------------------------------------------------------
# Derived panel geometry
# ---------------------------------------------------------------------------


def test_coordinate_token_geometry_is_resolution_independent_and_untouched():
    row = {
        "image_id": 1,
        "file_name": "images/x.jpg",
        "width": 100,
        "height": 50,
        "images": ["a.jpg"],
        "objects": [{"bbox_2d": ["<|coord_206|>", "<|coord_148|>", "<|coord_857|>", "<|coord_875|>"]}],
    }
    derived = prepare.derive_treatment_panel_row(
        row, treatment_width=200, treatment_height=100, relative_image_path="media/x.jpg"
    )
    assert derived["width"] == 200 and derived["height"] == 100
    assert derived["images"] == ["media/x.jpg"]
    assert derived["objects"][0]["bbox_2d"] == row["objects"][0]["bbox_2d"]
    # The source row is not mutated in place.
    assert row["width"] == 100


def test_pixel_geometry_scales_and_stays_a_valid_clamped_box():
    row = {
        "width": 100,
        "height": 50,
        "images": ["a.jpg"],
        "objects": [
            {"bbox_2d": [10, 5, 90, 45]},
            # Degenerate at source, and on the right/bottom edge: must stay
            # strictly ordered and inside the new canvas.
            {"bbox_2d": [99, 49, 99, 49]},
            {"poly": [0, 0, 100, 0, 100, 50]},
        ],
    }
    derived = prepare.derive_treatment_panel_row(
        row, treatment_width=200, treatment_height=100, relative_image_path="media/x.jpg"
    )
    first, second, third = derived["objects"]
    assert first["bbox_2d"] == [20, 10, 180, 90]
    x1, y1, x2, y2 = second["bbox_2d"]
    assert x2 > x1 and y2 > y1
    assert 0 <= x1 < 200 and 0 <= x2 < 200 and 0 <= y1 < 100 and 0 <= y2 < 100
    assert all(0 <= value for value in third["poly"])
    assert max(third["poly"][0::2]) <= 199 and max(third["poly"][1::2]) <= 99


def test_mixed_and_unsupported_geometry_fails_closed():
    mixed = {
        "width": 10,
        "height": 10,
        "images": ["a.jpg"],
        "objects": [{"bbox_2d": ["<|coord_1|>", 2, 3, 4]}],
    }
    with pytest.raises(prepare.PrepareContractError):
        prepare.derive_treatment_panel_row(
            mixed, treatment_width=20, treatment_height=20, relative_image_path="m/x.jpg"
        )
    lined = {
        "width": 10,
        "height": 10,
        "images": ["a.jpg"],
        "objects": [{"line": [1, 2, 3, 4]}],
    }
    with pytest.raises(prepare.PrepareContractError):
        prepare.derive_treatment_panel_row(
            lined, treatment_width=20, treatment_height=20, relative_image_path="m/x.jpg"
        )


# ---------------------------------------------------------------------------
# Overlay application
# ---------------------------------------------------------------------------

IMAGE_ID = "6040"
CURRENT_PROMPT = [11, 12, 13, 14]
TREATMENT_PROMPT = [11, 12, 13, 14, 15, 16]
CATEGORY_TOKENS = [900]
GENERATED_PREFIX = [
    planner.OBJECT_REF_START,
    *CATEGORY_TOKENS,
    planner.OBJECT_REF_END,
    planner.BOX_START,
    planner.COORD_TOKEN_START,
    planner.COORD_TOKEN_START + 1,
    planner.COORD_TOKEN_START + 2,
    planner.COORD_TOKEN_START + 3,
    planner.BOX_END,
]


def build_registries() -> dict[str, Any]:
    suffix = planner.build_query_suffix(CATEGORY_TOKENS)
    images = {
        IMAGE_ID: {
            "image_id": IMAGE_ID,
            "image_width": 1344,
            "image_height": 736,
            "prompt_token_ids": list(CURRENT_PROMPT),
            "prompt_token_ids_sha256": planner.sha256_json(CURRENT_PROMPT),
            "executed_media_sha256": "current-media",
            "wrapper_token_ids": dict(planner.WRAPPER_TOKEN_IDS),
            "coordinate_token_ids": {
                "start": planner.COORD_TOKEN_START,
                "end_inclusive": planner.COORD_TOKEN_END,
                "bin_count": planner.COORD_BIN_COUNT,
            },
        }
    }
    contexts = {}
    query_groups = {}
    for boundary_index, generated in enumerate(([], GENERATED_PREFIX)):
        context_id = f"{IMAGE_ID}:boundary-{boundary_index:03d}"
        observed = [*CURRENT_PROMPT, *generated]
        contexts[context_id] = {
            "context_id": context_id,
            "image_id": IMAGE_ID,
            "boundary_index": boundary_index,
            "generated_prefix_token_ids": list(generated),
            "generated_prefix_token_ids_sha256": planner.sha256_json(list(generated)),
            "prompt_token_ids_sha256": planner.sha256_json(CURRENT_PROMPT),
            "observed_self_prefix_token_ids_sha256": planner.sha256_json(observed),
            "observed_self_prefix_token_count": len(observed),
            "loop_marking": {"loop_tail": False},
        }
        observed_sha = planner.sha256_json(observed)
        query_prefix = [*observed, *suffix]
        query_sha = planner.sha256_json(query_prefix)
        group_id = f"{context_id}|person"
        query_groups[group_id] = {
            "query_group_id": group_id,
            "image_id": IMAGE_ID,
            "context_id": context_id,
            "category_query_id": f"{IMAGE_ID}:person",
            "normalized_description": "person",
            "status": "admitted",
            "query_suffix_token_ids": list(suffix),
            "query_suffix_token_ids_sha256": planner.sha256_json(suffix),
            "observed_prefix_sha256": observed_sha,
            "observed_prefix_token_count": len(observed),
            "query_prefix_sha256": query_sha,
            "query_prefix_token_count": len(query_prefix),
            "admission_receipt_id": planner.admission_receipt_id(
                context_id=context_id,
                channel=planner.CHANNEL_QUERY_SUFFIX,
                prefix_sha256=query_sha,
            ),
            "proposal_boundary_gate_admission_receipt_id": planner.admission_receipt_id(
                context_id=context_id,
                channel=planner.CHANNEL_PROPOSAL_BOUNDARY_GATE,
                prefix_sha256=observed_sha,
            ),
            "proposal_route_admission_receipt_id": (
                planner.proposal_route_admission_receipt_id(
                    context_id=context_id,
                    observed_prefix_sha256=observed_sha,
                    category_token_ids=CATEGORY_TOKENS,
                )
            ),
            "proposal_route_token_ids": planner.proposal_route_token_ids(CATEGORY_TOKENS),
            "proposal_route_digest": planner.proposal_route_digest(CATEGORY_TOKENS),
            "candidate_ids": ["cand:a", "cand:b"],
            "singleton_group_key": {
                "image_id": IMAGE_ID,
                "context_id": context_id,
                "normalized_description": "person",
                "observed_prefix_sha256": observed_sha,
                "query_prefix_sha256": query_sha,
            },
        }
    categories = {
        f"{IMAGE_ID}:person": {
            "category_query_id": f"{IMAGE_ID}:person",
            "category_token_ids": list(CATEGORY_TOKENS),
        }
    }
    candidates = {
        "cand:a": {
            "candidate_id": "cand:a",
            "coord_token_ids": [planner.COORD_TOKEN_START + i for i in range(4)],
            "coord_token_ids_sha256": "a",
            "strict_assignment_status": "matched",
            "strict_assignment_gt_owner_id": "gt:6040:0",
            "ambiguity_owner_ids": [],
        },
        "cand:b": {
            "candidate_id": "cand:b",
            "coord_token_ids": [planner.COORD_TOKEN_START + i for i in range(4, 8)],
            "coord_token_ids_sha256": "b",
            "strict_assignment_status": "unmatched",
            "strict_assignment_gt_owner_id": None,
            "ambiguity_owner_ids": [],
        },
    }
    owners = {
        "gt:6040:0": {
            "gt_owner_id": "gt:6040:0",
            "image_id": IMAGE_ID,
            "bbox_pixel_xyxy": [1.0, 2.0, 3.0, 4.0],
            "normalized_description": "person",
            "native_true_positive": False,
            "candidate_bank": {"physical_candidate_ids": ["cand:a", "cand:b"]},
        }
    }
    return {
        "images": images,
        "contexts": contexts,
        "query_groups": query_groups,
        "categories": categories,
        "candidates": candidates,
        "owners": owners,
    }


def build_overlay(**changes: Any) -> dict[str, Any]:
    overlay: dict[str, Any] = {
        "schema_version": prepare.OVERLAY_SCHEMA_VERSION,
        "intervention_unit_id": prepare.INTERVENTION_UNIT_ID,
        "arm_id": prepare.ARM_ID,
        "baseline_arm_id": prepare.BASELINE_ARM_ID,
        "max_pixels": prepare.TREATMENT_MAX_PIXELS,
        "base": {
            "plan_receipt_content_sha256": "base-receipt-digest",
            "capture_rules_sha256": "capture-rules-digest",
        },
        "treatment_panel": {"relative_path": prepare.TREATMENT_PANEL_NAME},
        "images": {
            IMAGE_ID: {
                "current": {
                    "prompt_token_ids_sha256": planner.sha256_json(CURRENT_PROMPT),
                    "prompt_token_count": len(CURRENT_PROMPT),
                    "executed_media_sha256": "current-media",
                    "merged_visual_tokens": 966,
                },
                "treatment": {
                    "width": 1920,
                    "height": 1056,
                    "executed_media_sha256": "treatment-media",
                    "prompt_token_ids": list(TREATMENT_PROMPT),
                    "prompt_token_ids_sha256": planner.sha256_json(TREATMENT_PROMPT),
                    "prompt_token_count": len(TREATMENT_PROMPT),
                    "merged_visual_tokens": 1980,
                    "image_grid_thw": [1, 66, 120],
                },
            }
        },
        "query_group_selection": {
            "query_group_ids_by_image": {IMAGE_ID: [f"{IMAGE_ID}:boundary-000|person"]}
        },
    }
    overlay.update(changes)
    overlay["overlay_content_sha256"] = prepare.overlay_content_sha256(overlay)
    return overlay


def test_overlay_swaps_the_prompt_and_reseals_prefix_identity():
    registries = build_registries()
    before = copy.deepcopy(registries)
    overlay = build_overlay()

    report = prepare.apply_overlay(overlay, **registries)
    assert report["frozen_content_preserved"] is True
    assert report["arm_id"] == prepare.ARM_ID

    image = registries["images"][IMAGE_ID]
    assert image["prompt_token_ids"] == TREATMENT_PROMPT
    assert image["prompt_token_ids_sha256"] == planner.sha256_json(TREATMENT_PROMPT)
    assert (image["image_width"], image["image_height"]) == (1920, 1056)
    assert image["executed_media_sha256"] == "treatment-media"

    for context_id, context in registries["contexts"].items():
        original = before["contexts"][context_id]
        # Generated prefix bytes survive verbatim.
        assert context["generated_prefix_token_ids"] == original["generated_prefix_token_ids"]
        assert (
            context["generated_prefix_token_ids_sha256"]
            == original["generated_prefix_token_ids_sha256"]
        )
        expected_observed = [*TREATMENT_PROMPT, *context["generated_prefix_token_ids"]]
        assert context["observed_self_prefix_token_ids_sha256"] == planner.sha256_json(
            expected_observed
        )
        assert context["observed_self_prefix_token_count"] == len(expected_observed)
        assert context["observed_self_prefix_token_ids_sha256"] != (
            original["observed_self_prefix_token_ids_sha256"]
        )

    for group_id, group in registries["query_groups"].items():
        original = before["query_groups"][group_id]
        # Frozen: suffix, candidates, routing identity.
        assert group["query_suffix_token_ids"] == original["query_suffix_token_ids"]
        assert group["query_suffix_token_ids_sha256"] == original["query_suffix_token_ids_sha256"]
        assert group["candidate_ids"] == original["candidate_ids"]
        assert group["proposal_route_token_ids"] == original["proposal_route_token_ids"]
        assert group["proposal_route_digest"] == original["proposal_route_digest"]
        # Changed: prefix digests and every exact-prefix admission id.
        assert group["observed_prefix_sha256"] != original["observed_prefix_sha256"]
        assert group["query_prefix_sha256"] != original["query_prefix_sha256"]
        assert group["admission_receipt_id"] != original["admission_receipt_id"]
        assert group["proposal_boundary_gate_admission_receipt_id"] != (
            original["proposal_boundary_gate_admission_receipt_id"]
        )
        assert group["proposal_route_admission_receipt_id"] != (
            original["proposal_route_admission_receipt_id"]
        )
        # And they are the digests of the literal treatment sequences.
        context = registries["contexts"][group["context_id"]]
        observed = [*TREATMENT_PROMPT, *context["generated_prefix_token_ids"]]
        query_prefix = [*observed, *group["query_suffix_token_ids"]]
        assert group["observed_prefix_sha256"] == planner.sha256_json(observed)
        assert group["query_prefix_sha256"] == planner.sha256_json(query_prefix)
        assert group["query_prefix_token_count"] == len(query_prefix)
        assert group["admission_receipt_id"] == planner.admission_receipt_id(
            context_id=group["context_id"],
            channel=planner.CHANNEL_QUERY_SUFFIX,
            prefix_sha256=group["query_prefix_sha256"],
        )
        assert group["singleton_group_key"]["query_prefix_sha256"] == (
            group["query_prefix_sha256"]
        )

    # Candidate bank, coordinate tokens and assignments are untouched.
    assert registries["candidates"] == before["candidates"]
    assert registries["owners"] == before["owners"]


def test_overlay_is_idempotent_under_reapplication():
    registries = build_registries()
    overlay = build_overlay()
    prepare.apply_overlay(overlay, **registries)
    first = copy.deepcopy(registries)
    # The overlay's "current" side no longer matches the plan, so a second
    # application must be refused rather than compounding a prompt swap.
    with pytest.raises(prepare.PrepareContractError):
        prepare.apply_overlay(overlay, **registries)
    assert registries["images"][IMAGE_ID] == first["images"][IMAGE_ID]


def test_overlay_refuses_a_null_intervention():
    registries = build_registries()
    overlay = build_overlay()
    block = overlay["images"][IMAGE_ID]["treatment"]
    block["prompt_token_ids"] = list(CURRENT_PROMPT)
    block["prompt_token_ids_sha256"] = planner.sha256_json(CURRENT_PROMPT)
    overlay["overlay_content_sha256"] = prepare.overlay_content_sha256(overlay)
    with pytest.raises(prepare.PrepareContractError, match="null intervention"):
        prepare.apply_overlay(overlay, **registries)


def test_overlay_refuses_a_tampered_seal_or_a_foreign_arm():
    overlay = build_overlay()
    overlay["max_pixels"] = 1
    with pytest.raises(prepare.PrepareContractError, match="reconstruct its own digest"):
        prepare.assert_overlay_seal(overlay)

    foreign = build_overlay(arm_id="some-other-arm")
    with pytest.raises(prepare.PrepareContractError, match="this unit executes"):
        prepare.assert_overlay_seal(foreign)


def test_overlay_refuses_a_prompt_digest_that_does_not_reconstruct():
    registries = build_registries()
    overlay = build_overlay()
    overlay["images"][IMAGE_ID]["treatment"]["prompt_token_ids_sha256"] = "0" * 64
    overlay["overlay_content_sha256"] = prepare.overlay_content_sha256(overlay)
    with pytest.raises(prepare.PrepareContractError, match="do not reconstruct"):
        prepare.apply_overlay(overlay, **registries)


def test_overlay_refuses_a_partial_image_set():
    registries = build_registries()
    registries["images"]["9999"] = dict(registries["images"][IMAGE_ID], image_id="9999")
    overlay = build_overlay()
    with pytest.raises(prepare.PrepareContractError, match="partial presentation swap"):
        prepare.apply_overlay(overlay, **registries)


def test_overlay_binding_checks_the_predecessor_plan_digest():
    overlay = build_overlay()
    prepare.assert_overlay_binds_plan(
        overlay,
        plan_receipt_content_sha256="base-receipt-digest",
        capture_rules_sha256="capture-rules-digest",
    )
    with pytest.raises(prepare.PrepareContractError, match="different predecessor plan"):
        prepare.assert_overlay_binds_plan(
            overlay, plan_receipt_content_sha256="another-digest"
        )
    with pytest.raises(prepare.PrepareContractError, match="different predecessor capture"):
        prepare.assert_overlay_binds_plan(
            overlay,
            plan_receipt_content_sha256="base-receipt-digest",
            capture_rules_sha256="other-rules",
        )


def test_overlay_round_trips_through_disk(tmp_path: Path):
    overlay = build_overlay()
    path = tmp_path / prepare.OVERLAY_NAME
    path.write_bytes(planner.canonical_json_bytes(overlay) + b"\n")
    assert prepare.load_overlay(path)["overlay_content_sha256"] == (
        overlay["overlay_content_sha256"]
    )

    edited = json.loads(path.read_text())
    edited["images"][IMAGE_ID]["treatment"]["merged_visual_tokens"] = 1
    path.write_bytes(planner.canonical_json_bytes(edited) + b"\n")
    with pytest.raises(prepare.PrepareContractError):
        prepare.load_overlay(path)


# ---------------------------------------------------------------------------
# Query-group selection
# ---------------------------------------------------------------------------


class _StubPlan:
    def __init__(self, *, contexts, query_groups, owners, native_sidecars):
        self.contexts = contexts
        self.query_groups = query_groups
        self.owners = owners
        self.native_sidecars = native_sidecars


def _selection_plan() -> _StubPlan:
    contexts = {}
    query_groups = {}
    for image_id, boundary_count in (("7511", 3), (prepare.RESTRICTED_IMAGE_ID, 4)):
        for index in range(boundary_count):
            context_id = f"{image_id}:boundary-{index:03d}"
            # Boundary 002 of each image is a loop tail.
            contexts[context_id] = {
                "context_id": context_id,
                "image_id": image_id,
                "boundary_index": index,
                "loop_marking": {"loop_tail": index == 2},
            }
            for description in ("person", "book"):
                group_id = f"{context_id}|{description}"
                query_groups[group_id] = {
                    "query_group_id": group_id,
                    "image_id": image_id,
                    "context_id": context_id,
                    "normalized_description": description,
                    "status": "admitted",
                    "candidate_ids": ["cand:1", "cand:2"],
                }
    owners = {
        "gt:7511:tp": {
            "gt_owner_id": "gt:7511:tp",
            "image_id": "7511",
            "native_true_positive": True,
            "native_strict_match_pred_row_ids": ["pred:1"],
        }
    }
    native_sidecars = [{"pred_row_id": "pred:1", "row_index": 1}]
    return _StubPlan(
        contexts=contexts,
        query_groups=query_groups,
        owners=owners,
        native_sidecars=native_sidecars,
    )


def _summary(
    *,
    owner_id: str,
    image_id: str,
    description: str,
    disposition: str,
    split: str = "discovery",
    native_true_positive: bool = False,
    usable_support_context_ids: list[str] | None = None,
    frontier_context: str | None = None,
    diagnostic_context: str | None = None,
) -> dict[str, Any]:
    return {
        "gt_owner_id": owner_id,
        "image_id": image_id,
        "normalized_description": description,
        "split": split,
        "disposition": disposition,
        "native_true_positive": native_true_positive,
        "upper_bound_u": {
            "usable_support": disposition == prepare.RESOLVED_DISPOSITION,
            "usable_support_context_ids": list(usable_support_context_ids or []),
            "primary_first_non_loop_minimal_abs_frontier": (
                None if frontier_context is None else {"context_id": frontier_context}
            ),
            "diagnostic_best_all": (
                None if diagnostic_context is None else {"context_id": diagnostic_context}
            ),
        },
    }


def test_selection_takes_whole_categories_on_non_restricted_images():
    plan = _selection_plan()
    summaries = [
        _summary(
            owner_id="gt:7511:0",
            image_id="7511",
            description="person",
            disposition=prepare.PERSISTENT_DISPOSITION,
        )
    ]
    selection = prepare.select_query_groups(plan, summaries)
    groups = set(selection["query_group_ids_by_image"]["7511"])
    # Every non-loop context of the persistent owner's category, and no other
    # category, and never the loop tail.
    assert groups == {"7511:boundary-000|person", "7511:boundary-001|person"}
    assert selection["cohort_owner_counts"]["persistent_outside_restricted"] == 1
    assert selection["cost"]["candidate_score_rows"] == 4


def test_selection_restricts_the_restricted_image_to_a_declared_context_union():
    plan = _selection_plan()
    restricted = prepare.RESTRICTED_IMAGE_ID
    summaries = [
        _summary(
            owner_id=f"gt:{restricted}:0",
            image_id=restricted,
            description="person",
            disposition=prepare.PERSISTENT_DISPOSITION,
            split="confirmation",
            frontier_context=f"{restricted}:boundary-001",
            diagnostic_context=f"{restricted}:boundary-003",
        )
    ]
    selection = prepare.select_query_groups(plan, summaries)
    assert set(selection["query_group_ids_by_image"][restricted]) == {
        f"{restricted}:boundary-000|person",
        f"{restricted}:boundary-001|person",
        f"{restricted}:boundary-003|person",
    }
    assert selection["cohort_owner_counts"]["persistent_restricted"] == 1
    assert selection["cohort_owner_counts"]["persistent_outside_restricted"] == 0


def test_selection_covers_calibration_and_retention_controls():
    plan = _selection_plan()
    summaries = [
        _summary(
            owner_id="gt:7511:tp",
            image_id="7511",
            description="person",
            disposition="native_true_positive_calibration_control",
            native_true_positive=True,
        ),
        _summary(
            owner_id="gt:7511:res",
            image_id="7511",
            description="book",
            disposition=prepare.RESOLVED_DISPOSITION,
            usable_support_context_ids=["7511:boundary-000"],
            frontier_context="7511:boundary-001",
        ),
    ]
    selection = prepare.select_query_groups(plan, summaries)
    groups = set(selection["query_group_ids_by_image"]["7511"])
    # The discovery TP goes to its deterministic due context (row_index 1).
    assert "7511:boundary-001|person" in groups
    assert selection["query_group_reasons"]["7511:boundary-001|person"] == [
        "discovery_tp_calibration"
    ]
    # The resolved owner goes to its frozen usable support and frontier contexts.
    assert {"7511:boundary-000|book", "7511:boundary-001|book"} <= groups
    assert selection["cohort_owner_counts"]["discovery_tp_calibration"] == 1
    assert selection["cohort_owner_counts"]["resolved_retention"] == 1


def test_selection_never_touches_a_treatment_score():
    plan = _selection_plan()
    summaries = [
        _summary(
            owner_id="gt:7511:0",
            image_id="7511",
            description="person",
            disposition=prepare.PERSISTENT_DISPOSITION,
        )
    ]
    selection = prepare.select_query_groups(plan, summaries)
    assert selection["selection_policy"].endswith("never_score_selected")
    with pytest.raises(prepare.PrepareContractError, match="selection is empty"):
        prepare.select_query_groups(plan, [])


# ---------------------------------------------------------------------------
# Digest helpers
# ---------------------------------------------------------------------------


def test_executed_media_digest_matches_the_production_definition(tmp_path: Path):
    from PIL import Image

    image = Image.new("RGB", (4, 3), color=(1, 2, 3))
    digest = hashlib.sha256()
    digest.update(b"coordexp-rgb8-pixels-v1\0")
    digest.update((4).to_bytes(8, "big"))
    digest.update((3).to_bytes(8, "big"))
    digest.update(image.tobytes())
    assert prepare.rgb_pixel_sha256(image) == digest.hexdigest()
