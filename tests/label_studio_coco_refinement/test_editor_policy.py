from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pytest

from src.common.errors import DataContractError
from src.label_studio_coco_refinement.categories import COCO80_REGISTRY
from src.label_studio_coco_refinement import editor_policy
from src.label_studio_coco_refinement.editor_policy import (
    ACCESSIBLE_HIGH_CONTRAST_PALETTE,
    EDITOR_POLICY_GOLDEN_VECTORS_SHA256,
    VISUAL_POLICY_ID,
    EditorRegion,
    MatchKind,
    canonical_class_payload,
    derive_visual_policy,
    editor_policy_golden_vectors,
    search_coco80_classes,
)


def _region(
    key: str,
    bbox: tuple[int, int, int, int],
    *,
    category: str = "person",
    inferred: bool = True,
) -> EditorRegion:
    return EditorRegion(
        stable_region_key=key,
        canonical_name=category,
        bbox_2d=bbox,
        is_uncommitted_inference=inferred,
    )


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _region_from_vector(value: dict[str, object]) -> EditorRegion:
    return EditorRegion(
        stable_region_key=value["stable_region_key"],
        canonical_name=value["canonical_name"],
        bbox_2d=tuple(value["bbox_2d"]),
        is_uncommitted_inference=value["is_uncommitted_inference"],
    )


def test_search_normalizes_case_and_repeated_whitespace_but_returns_canonical() -> None:
    results = search_coco80_classes("  TRAFFIC   light ")

    assert results == (results[0],)
    assert results[0].canonical_name == "traffic light"
    assert results[0].category_id == 10
    assert results[0].match_kind is MatchKind.EXACT
    assert results[0].spelling_distance is None
    assert canonical_class_payload(results[0].canonical_name) == {
        "desc": "traffic light",
        "category_name": "traffic light",
        "category_id": 10,
    }


def test_search_priority_and_stable_ties_are_keyboard_consumable() -> None:
    car_results = search_coco80_classes("car", limit=3)
    board_results = search_coco80_classes("board", limit=4)

    assert [
        (item.rank, item.canonical_name, item.match_kind) for item in car_results[:2]
    ] == [
        (0, "car", MatchKind.EXACT),
        (1, "carrot", MatchKind.PREFIX),
    ]
    assert [item.canonical_name for item in board_results] == [
        "snowboard",
        "skateboard",
        "surfboard",
        "keyboard",
    ]
    assert all(item.match_kind is MatchKind.SUBSTRING for item in board_results)


@pytest.mark.parametrize("query", ["trafik", "trafic light"])
def test_traffic_light_remains_selectable_for_small_typos(query: str) -> None:
    results = search_coco80_classes(query, limit=3)

    assert results[0].canonical_name == "traffic light"
    assert results[0].match_kind is MatchKind.SPELLING
    assert results[0].spelling_distance in {1, 2}


@pytest.mark.parametrize("query", ["交通灯", "signal lamp", "automobile", "spaceship"])
def test_translation_synonym_and_free_text_never_become_output(query: str) -> None:
    assert search_coco80_classes(query) == ()
    with pytest.raises(DataContractError, match="exact canonical"):
        canonical_class_payload(query)


def test_every_search_result_is_canonical_and_repeated_calls_are_identical() -> None:
    expected = search_coco80_classes("bseball", limit=10)

    assert search_coco80_classes("bseball", limit=10) == expected
    assert all(
        COCO80_REGISTRY.by_name(result.canonical_name).id == result.category_id
        for result in expected
    )


def test_expanded_neighbor_intersection_includes_touching_boundary_and_clips_edges() -> (
    None
):
    result = derive_visual_policy(
        (
            _region("edge", (0, 0, 10, 10)),
            _region("touch", (34, 0, 50, 10)),
            _region("gap", (35, 100, 50, 110)),
            _region("far", (999 - 10, 999 - 10, 999, 999)),
        )
    )

    assert result.neighbor_pairs == (("edge", "touch"),)
    colors = {item.stable_region_key: item.color for item in result.presentations}
    assert colors["edge"] != colors["touch"]


def test_visual_policy_is_invariant_to_input_reordering() -> None:
    regions = (
        _region("z", (100, 100, 200, 200), category="car"),
        _region("a", (190, 100, 300, 200), category="car"),
        _region("m", (700, 700, 800, 800), category="person"),
    )

    forward = derive_visual_policy(regions).to_json_dict()
    reverse = derive_visual_policy(reversed(regions)).to_json_dict()

    assert forward == reverse
    assert [item["stable_region_key"] for item in forward["presentations"]] == [
        "a",
        "m",
        "z",
    ]


def test_palette_exhaustion_reuses_deterministically_with_numeric_badges() -> None:
    count = len(ACCESSIBLE_HIGH_CONTRAST_PALETTE) + 2
    regions = tuple(
        _region(f"region-{index:02d}", (100, 100, 300, 300)) for index in range(count)
    )

    result = derive_visual_policy(regions)

    assert len(result.neighbor_pairs) == count * (count - 1) // 2
    assert [item.palette_index for item in result.presentations[:8]] == list(range(8))
    assert result.presentations[8].palette_index == 0
    assert result.presentations[8].numeric_badge == 9
    assert result.presentations[9].palette_index == 1
    assert result.presentations[9].numeric_badge == 10
    assert derive_visual_policy(reversed(regions)) == result


def test_duplicate_cues_use_same_canonical_class_and_exact_half_iou_threshold() -> None:
    regions = (
        _region("base", (0, 0, 100, 100), inferred=False),
        _region("half", (0, 0, 100, 50)),
        _region("below", (0, 0, 100, 49)),
        _region("other-class", (0, 0, 100, 100), category="car"),
    )

    result = derive_visual_policy(regions)

    assert ("base", "half") in result.duplicate_pairs
    assert ("base", "below") not in result.duplicate_pairs
    assert all("other-class" not in pair for pair in result.duplicate_pairs)
    assert [item.stable_region_key for item in result.presentations] == [
        "below",
        "half",
        "other-class",
    ]


def test_advisory_visual_output_does_not_mutate_or_embed_semantic_payload() -> None:
    semantic_payload = {
        "bbox_2d": [100, 100, 300, 300],
        "desc": "person",
        "category_name": "person",
        "category_id": 1,
        "coco_ann_id": -1,
    }
    before = deepcopy(semantic_payload)

    output = derive_visual_policy(
        (_region("infer", tuple(semantic_payload["bbox_2d"])),)
    ).to_json_dict()

    assert semantic_payload == before
    assert output["policy_id"] == VISUAL_POLICY_ID
    serialized = json.dumps(output, sort_keys=True)
    for semantic_field in ("bbox_2d", "desc", "category_id", "coco_ann_id"):
        assert semantic_field not in serialized


def test_golden_fixture_is_literal_json_and_has_complete_sorted_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("golden expected must not call policy implementations")

    monkeypatch.setattr(editor_policy, "search_coco80_classes", forbidden)
    monkeypatch.setattr(editor_policy, "derive_visual_policy", forbidden)
    vectors = editor_policy_golden_vectors()

    assert json.loads(json.dumps(vectors, ensure_ascii=True, sort_keys=True)) == vectors
    assert _canonical_json_sha256(vectors) == EDITOR_POLICY_GOLDEN_VECTORS_SHA256
    assert vectors["schema_version"] == "coordexp-editor-policy-golden-v1"


def test_search_matches_independent_golden_ties_and_normalization() -> None:
    vectors = editor_policy_golden_vectors()

    for vector in vectors["search"]:
        observed = search_coco80_classes(**vector["input"])
        assert [result.to_json_dict() for result in observed] == vector["expected"]

    by_name = {vector["name"]: vector for vector in vectors["search"]}
    assert [
        result["canonical_name"]
        for result in by_name["spelling_distance_prefix_tie"]["expected"]
    ] == ["traffic light", "train"]
    assert by_name["normalization_exact"]["expected"][0]["match_kind"] == "exact"


def test_visual_policy_matches_independent_golden_boundaries_palette_and_iou() -> None:
    vectors = editor_policy_golden_vectors()
    by_name = {vector["name"]: vector for vector in vectors["visual_policy"]}

    for vector in vectors["visual_policy"]:
        regions = tuple(_region_from_vector(value) for value in vector["input"])
        observed = derive_visual_policy(regions).to_json_dict()
        expected = vector["expected"]
        assert _canonical_json_sha256(observed) == expected["output_sha256"]
        assert observed["palette"] == expected["palette"]
        assert observed["presentations"] == expected["presentations"]
        assert observed["duplicate_pairs"] == expected["duplicate_pairs"]
        if "neighbor_pairs" in expected:
            assert observed["neighbor_pairs"] == expected["neighbor_pairs"]
        else:
            assert len(observed["neighbor_pairs"]) == expected["neighbor_pair_count"]
            assert (
                _canonical_json_sha256(observed["neighbor_pairs"])
                == expected["neighbor_pairs_sha256"]
            )

    assert by_name["neighbor_gap_24_bins"]["expected"]["neighbor_pairs"] == [
        ["left", "right"]
    ]
    assert by_name["neighbor_gap_25_bins"]["expected"]["neighbor_pairs"] == []
    clique = by_name["palette_clique_exhaustion"]["expected"]
    assert clique["palette"] == list(ACCESSIBLE_HIGH_CONTRAST_PALETTE)
    assert [item["color"] for item in clique["presentations"]] == [
        *ACCESSIBLE_HIGH_CONTRAST_PALETTE,
        ACCESSIBLE_HIGH_CONTRAST_PALETTE[0],
        ACCESSIBLE_HIGH_CONTRAST_PALETTE[1],
    ]
    assert [item["numeric_badge"] for item in clique["presentations"][-2:]] == [9, 10]
    assert by_name["duplicate_exact_iou_half"]["expected"]["duplicate_pairs"] == [
        ["base", "half"]
    ]


def test_browser_visual_policy_fixture_is_an_exact_parent_golden_projection() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    browser_fixture = json.loads(
        (
            repo_root
            / "label-studio/web/libs/editor/src/components/CoordExpAIRegion/visual-policy.golden.json"
        ).read_text(encoding="utf-8")
    )
    parent_fixture = editor_policy_golden_vectors()

    assert browser_fixture["schema_version"] == parent_fixture["schema_version"]
    parent_by_name = {
        vector["name"]: vector for vector in parent_fixture["visual_policy"]
    }
    assert {vector["name"] for vector in browser_fixture["visual_policy"]} == set(
        parent_by_name
    )
    for browser_vector in browser_fixture["visual_policy"]:
        parent_vector = parent_by_name[browser_vector["name"]]
        assert browser_vector["input"] == parent_vector["input"]
        for field, expected in browser_vector["expected"].items():
            assert parent_vector["expected"][field] == expected


def test_neighbor_pairs_are_unique_and_stably_ordered() -> None:
    regions = tuple(_region(key, (100, 100, 200, 200)) for key in ("c", "a", "b"))

    assert derive_visual_policy(regions).neighbor_pairs == tuple(
        combinations(("a", "b", "c"), 2)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stable_region_key": "", "canonical_name": "person", "bbox_2d": (0, 0, 1, 1)},
        {"stable_region_key": "x", "canonical_name": "Person", "bbox_2d": (0, 0, 1, 1)},
        {"stable_region_key": "x", "canonical_name": "person", "bbox_2d": (0, 0, 0, 1)},
        {
            "stable_region_key": "x",
            "canonical_name": "person",
            "bbox_2d": (0, 0, 1, 1000),
        },
    ],
)
def test_visual_policy_input_fails_closed_on_noncanonical_or_invalid_regions(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((DataContractError, ValueError)):
        EditorRegion(is_uncommitted_inference=True, **kwargs)
