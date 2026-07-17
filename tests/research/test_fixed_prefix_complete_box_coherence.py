from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from scripts.research.run_fixed_prefix_complete_box_coherence import (
    CLASSIFICATION_MARGIN,
    CLASSIFICATION_RADIUS,
    COORDINATE_TOKEN_START,
    _row_from_bundle,
    build_coordinate_release_arms,
    build_parser,
    box_shape_metadata,
    classifier_references_for_panel,
    classify_released_box,
    coordinate_bin,
    coordinate_token_id,
    deduplicate_panel_paths,
    enumerate_binary_hybrids,
    enumerate_target_panels,
    load_verified_fixture,
    main,
    materialize_fixture,
    paired_release_sampling_seeds,
    parse_coordinate_release_suffix,
    score_coordinate_path,
    score_primary_coordinate_path,
    select_release_panel,
    summarize_released_owner_crossover,
    run_progressive_coordinate_release,
)


BUNDLE_A = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/executions/"
    "dense-union-51-primary-after-wave-local-tail-contract/artifacts/calls/"
    "1db840709abcf5e20a5bd475fa1acfbeca2bcf64060374b856218444337f6762/"
    "terminal-output-bundle.json"
)
BUNDLE_C = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/executions/"
    "dense-union-51-primary-after-wave-local-tail-contract/artifacts/calls/"
    "2b4c85bbd63fe017d8c9b67028e46c4f60006c93cf88974cff010c47cf0a200f/"
    "terminal-output-bundle.json"
)
SOURCE_JSONL = Path(
    "/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/"
    "val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl"
)


def test_exact_row_and_pre_x1_fixture_anchors() -> None:
    fixture = materialize_fixture(source_jsonl=SOURCE_JSONL, bundles={"A": BUNDLE_A, "C": BUNDLE_C})
    assert set(fixture["targets"]) == {"A", "C"}
    target_a = fixture["targets"]["A"]
    target_c = fixture["targets"]["C"]
    assert target_a["request_id"].endswith("18404482ab0b688758cca6759b5f83bbe5e5959d3689641fdc53bf857977d8a2")
    assert target_c["request_id"].endswith("ff5660928aa3d9401fb9679d0112b3a5477f1bf3b13a25b9339c2911e02fa0d9")
    assert target_a["row"]["source_row_coordinates"] == [658, 450, 780, 487]
    assert target_a["references"][0]["box"] == target_a["row"]["source_row_coordinates"]
    assert len(target_a["row"]["row_token_ids"]) == 9
    assert target_a["row"]["row_token_ids"][0] == 151646
    assert target_c["row"]["source_row_coordinates"] == [552, 123, 660, 357]
    assert target_c["diagnostics"]["source_row4_union_diagnostic"]["source_row_coordinates"] == [519, 33, 657, 347]
    assert target_a["row"]["pre_x1_prompt_token_ids_sha256"]
    assert len(target_a["row"]["pre_x1_prompt_token_ids"]) == 1426
    assert target_c["row"]["pre_x1_prompt_token_ids_sha256"]
    assert all(max(ref["round_trip"]["absolute_bin_errors"]) <= 1 for ref in target_c["references"])


def test_row_extraction_rejects_changed_coordinate_name() -> None:
    bundle = json.loads(BUNDLE_A.read_text(encoding="utf-8"))
    bundle["decode_result"]["token_trace"][106]["token_text"] = "<|coord_659|>"
    with pytest.raises(ValueError, match="token id/name"):
        _row_from_bundle(bundle, 11)


def test_hybrid_enumeration_has_unique_endpoints_and_shape_metadata() -> None:
    hybrids = enumerate_binary_hybrids([10, 20, 80, 90], [30, 40, 60, 70], prefix="pair")
    assert len(hybrids) == 16
    assert len({tuple(item["box"]) for item in hybrids}) == 16
    assert len({item["label"] for item in hybrids}) == 16
    assert sum(item["is_endpoint"] for item in hybrids) == 2
    assert sum(item["kind"] == "hybrid" for item in hybrids) == 14
    assert all(item["valid"] for item in hybrids)
    assert box_shape_metadata([10, 20, 80, 90])["area"] == 4900
    panels = enumerate_target_panels("C")
    assert panels[2]["panel"] == "target_vs_multi_instance_union"
    exact_hybrid = next(item for item in panels[0]["hybrids"] if item["kind"] == "hybrid")
    classifier_references = [
        *panels[0]["references"],
        *[item for item in panels[0]["hybrids"] if not item["is_endpoint"]],
    ]
    exact_hybrid_result = classify_released_box(exact_hybrid["box"], classifier_references)
    assert (
        exact_hybrid_result["boundary_configuration_attribution"]["label"]
        == "boundary_hybrid"
    )
    assert panels[0]["claim_contract"]["owner_claim_admitted"] is True
    assert panels[0]["claim_contract"]["earliest_semantically_meaningful_differing_slot"] == "x1"
    assert panels[1]["claim_contract"]["earliest_semantically_meaningful_differing_slot"] == "y1"
    assert panels[2]["claim_contract"]["owner_claim_admitted"] is False


def test_primary_score_uses_full_vocab_and_keeps_coordinate_normalization_diagnostic() -> None:
    vocab = 153000
    rows = []
    for selected_bin, full_bias in zip([10, 20, 30, 40], [2.0, 3.0, 4.0, 5.0]):
        logits = torch.full((vocab,), -50.0, dtype=torch.float32)
        logits[coordinate_token_id(selected_bin)] = full_bias
        logits[COORDINATE_TOKEN_START + 999] = full_bias + 0.1
        rows.append(logits)
    result = score_coordinate_path(rows, [10, 20, 30, 40])
    assert result["joint_primary_score_float32"] == pytest.approx(
        score_primary_coordinate_path(result["slot_summaries"])
    )
    assert all(len(slot["coordinate_logits_float32"]) == 1000 for slot in result["slot_summaries"])
    assert all("selected_full_vocabulary_logprob_float32" in slot for slot in result["slot_summaries"])
    assert all("selected_coordinate_normalized_logprob_float32" in slot for slot in result["slot_summaries"])
    assert all(len(slot["top20"]) == 20 for slot in result["slot_summaries"])


def test_release_arms_append_coordinate_tokens_without_repetition_penalty() -> None:
    prefix = [7, 8, 9]
    arms = build_coordinate_release_arms(prefix, [100, 200, 300, 400])
    assert [arm["forced_coordinate_count"] for arm in arms] == [0, 1, 2, 3, 4]
    assert arms[0]["prompt_token_ids"] == prefix
    assert arms[2]["prompt_token_ids"] == prefix + [coordinate_token_id(100), coordinate_token_id(200)]
    assert arms[4]["released_coordinate_slots"] == []
    assert all(arm["repetition_penalty"] == 1.0 for arm in arms)
    assert coordinate_bin(coordinate_token_id(999)) == 999


@pytest.mark.parametrize("forced_count", range(5))
def test_release_suffix_parser_completes_every_progressive_arm(forced_count: int) -> None:
    box = [100, 200, 700, 800]
    box_end = 151649
    generated = [
        *[coordinate_token_id(value) for value in box[forced_count:]],
        box_end,
        12345,
    ]
    parsed = parse_coordinate_release_suffix(
        generated,
        forced_coordinate_bins=box[:forced_count],
        box_end_token_id=box_end,
    )
    assert parsed["parser_valid"] is True
    assert parsed["completed_coordinate_bins"] == box
    assert parsed["box_end_generated_index"] == 4 - forced_count
    assert parsed["trailing_generated_token_ids"] == [12345]


def test_release_suffix_parser_distinguishes_token_closure_and_geometry_failures() -> None:
    non_coordinate = parse_coordinate_release_suffix(
        [42, 151649], forced_coordinate_bins=[100, 200, 700], box_end_token_id=151649
    )
    assert non_coordinate["failure"] == "non_coordinate_at_released_slot_3"
    missing_close = parse_coordinate_release_suffix(
        [coordinate_token_id(800), 42],
        forced_coordinate_bins=[100, 200, 700],
        box_end_token_id=151649,
    )
    assert missing_close["failure"] == "missing_natural_box_closure"
    invalid_geometry = parse_coordinate_release_suffix(
        [151649], forced_coordinate_bins=[700, 200, 100, 800], box_end_token_id=151649
    )
    assert invalid_geometry["failure"] == "invalid_completed_geometry"


def test_release_pair_selection_and_classifier_reference_freeze() -> None:
    target = {"panels": enumerate_target_panels("C")}
    panel, endpoints = select_release_panel(
        target,
        "target_vs_adjacent_owner",
        path_labels=["target_owner", "adjacent_owner"],
    )
    assert [path["label"] for path in endpoints] == ["target_owner", "adjacent_owner"]
    references = classifier_references_for_panel(panel)
    assert len(references) == 16
    assert sum(reference["kind"] == "hybrid" for reference in references) == 14
    assert panel["claim_contract"]["owner_discriminative_slots"] == ["x1", "x2"]
    upper = enumerate_target_panels("C")[1]
    union = enumerate_target_panels("C")[2]
    target_a = enumerate_target_panels("A")[0]
    assert upper["claim_contract"]["owner_discriminative_slots"] == ["y1", "y2"]
    assert union["claim_contract"]["owner_discriminative_slots"] == []
    assert union["claim_contract"]["region_discriminative_slots"] == ["x1", "y1"]
    assert target_a["claim_contract"]["owner_discriminative_slots"] == []
    assert target_a["claim_contract"]["extent_discriminative_slots"] == ["x2"]
    with pytest.raises(ValueError, match="not coherent endpoints"):
        select_release_panel(
            target,
            "target_vs_adjacent_owner",
            path_labels=["target_adjacent_owner:0001"],
        )


def test_paired_release_seeds_are_reproducible_unique_and_signed_64_bit() -> None:
    left = paired_release_sampling_seeds(17, 16)
    right = paired_release_sampling_seeds(17, 16)
    assert left == right
    assert len(set(left)) == 16
    assert all(0 <= seed < (1 << 63) - 1 for seed in left)


def test_release_parser_accepts_explicit_max_new_tokens() -> None:
    args = build_parser().parse_args(
        [
            "--phase",
            "release",
            "--max-new-tokens",
            "512",
        ]
    )
    assert args.max_new_tokens == 512


def test_release_parser_accepts_repeated_forced_coordinate_counts() -> None:
    args = build_parser().parse_args(
        [
            "--phase",
            "release",
            "--forced-coordinate-count",
            "1",
            "--forced-coordinate-count",
            "2",
        ]
    )
    assert args.forced_coordinate_count == [1, 2]


@pytest.mark.parametrize("max_new_tokens", [0, 1, 4])
def test_release_runtime_rejects_insufficient_max_new_tokens_before_model_load(
    max_new_tokens: int,
) -> None:
    fixture = {
        "targets": {
            "C": {
                "panels": enumerate_target_panels("C"),
            }
        }
    }
    with pytest.raises(ValueError, match="must be at least 5"):
        run_progressive_coordinate_release(
            fixture,
            target="C",
            panel_name="target_vs_adjacent_owner",
            max_new_tokens=max_new_tokens,
        )


def test_release_main_rejects_bad_fixture_digest_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "schema_version": "fixed_prefix_complete_box_coherence.fixture.v2",
        "targets": {},
    }
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    called = False

    def forbidden_runtime(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("runtime must not be reached")

    monkeypatch.setattr(
        "scripts.research.run_fixed_prefix_complete_box_coherence.run_progressive_coordinate_release",
        forbidden_runtime,
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        main(
            [
                "--phase",
                "release",
                "--fixture",
                str(fixture),
                "--fixture-sha256",
                "0" * 64,
                "--target",
                "C",
                "--panel",
                "target_vs_adjacent_owner",
                "--output",
                str(tmp_path / "release.json"),
            ]
        )
    assert called is False


def test_classifier_precedence_radius_margin_and_parser_gate() -> None:
    references = [
        {
            "label": "target_owner",
            "kind": "visible",
            "box": [100, 100, 300, 300],
            "is_endpoint": True,
        },
        {
            "label": "adjacent_owner",
            "kind": "visible",
            "box": [500, 100, 700, 300],
            "is_endpoint": True,
        },
        {
            "label": "hybrid:0101",
            "kind": "hybrid",
            "box": [100, 100, 700, 300],
            "is_endpoint": False,
        },
    ]
    target = classify_released_box([100, 100, 300, 300], references)
    adjacent = classify_released_box([500, 100, 700, 300], references)
    assert target["boundary_configuration_attribution"]["label"] == "target_owner"
    assert target["endpoint_family_attribution"]["label"] == "target_owner"
    assert adjacent["boundary_configuration_attribution"]["label"] == "adjacent_owner"
    assert adjacent["endpoint_family_attribution"]["label"] == "adjacent_owner"
    exact_hybrid = classify_released_box([100, 100, 700, 300], references)
    assert exact_hybrid["boundary_configuration_attribution"]["label"] == "boundary_hybrid"
    assert (
        exact_hybrid["boundary_configuration_attribution"]["precedence"]
        == "exact_nonendpoint_hybrid"
    )
    invalid = classify_released_box([100, 100, 300, 300], references, parser_valid=False)
    assert (
        invalid["boundary_configuration_attribution"]["label"]
        == "invalid_or_no_closure"
    )
    assert invalid["endpoint_family_attribution"]["label"] == "invalid_or_no_closure"
    assert CLASSIFICATION_RADIUS == 0.35
    assert CLASSIFICATION_MARGIN == 0.05


def test_classifier_exact_endpoint_precedes_near_one_bin_lattice_reference() -> None:
    references = [
        {
            "label": "target_owner",
            "kind": "visible",
            "box": [100, 100, 300, 300],
            "is_endpoint": True,
        },
        {
            "label": "adjacent_owner",
            "kind": "visible",
            "box": [500, 100, 700, 300],
            "is_endpoint": True,
        },
        {
            "label": "near_target_hybrid",
            "kind": "hybrid",
            "box": [101, 100, 300, 300],
            "is_endpoint": False,
        },
    ]
    result = classify_released_box([100, 100, 300, 300], references)
    boundary = result["boundary_configuration_attribution"]
    assert boundary["label"] == "target_owner"
    assert boundary["accepted"] is True
    assert boundary["precedence"] == "exact_coherent_endpoint"


def test_endpoint_family_midpoint_is_ambiguous_and_axis_disagreement_is_retained() -> None:
    references = [
        {
            "label": "target_owner",
            "kind": "visible",
            "box": [100, 100, 300, 300],
            "is_endpoint": True,
        },
        {
            "label": "adjacent_owner",
            "kind": "visible",
            "box": [300, 100, 500, 300],
            "is_endpoint": True,
        },
        {
            "label": "near_target_hybrid",
            "kind": "hybrid",
            "box": [101, 100, 300, 300],
            "is_endpoint": False,
        },
    ]
    midpoint = classify_released_box([200, 100, 400, 300], references)
    midpoint_endpoint = midpoint["endpoint_family_attribution"]
    assert midpoint_endpoint["label"] == "ambiguous_between_endpoint_families"
    assert midpoint_endpoint["accepted"] is False
    assert midpoint_endpoint["ambiguous"] is True
    assert midpoint_endpoint["outside"] is False

    disagreement = classify_released_box(
        [101, 100, 300, 300],
        references,
        forced_coordinate_count=1,
    )
    assert (
        disagreement["boundary_configuration_attribution"]["label"]
        == "boundary_hybrid"
    )
    assert disagreement["endpoint_family_attribution"]["label"] == "target_owner"
    edges = disagreement["slot_edge_comparisons"]
    assert edges[0]["generation_role"] == "forced"
    assert all(edge["generation_role"] == "released" for edge in edges[1:])
    assert edges[0]["absolute_error_to_left_endpoint"] == 1.0
    assert edges[0]["endpoint_separation"] == 200.0


def test_owner_crossover_uses_only_declared_adjacent_owner_slots() -> None:
    records = [
        {
            "path": {"label": "target_owner"},
            "arm": {"forced_coordinate_count": 1},
            "classification": {
                "slot_edge_comparisons": [
                    {
                        "slot": "x1",
                        "generation_role": "forced",
                        "endpoint_separation": 100.0,
                        "closer_endpoint": "target_owner",
                    },
                    {
                        "slot": "y1",
                        "generation_role": "released",
                        "endpoint_separation": 5.0,
                        "closer_endpoint": "adjacent_owner",
                    },
                    {
                        "slot": "x2",
                        "generation_role": "released",
                        "endpoint_separation": 80.0,
                        "closer_endpoint": "adjacent_owner",
                    },
                    {
                        "slot": "y2",
                        "generation_role": "released",
                        "endpoint_separation": 1.0,
                        "closer_endpoint": "adjacent_owner",
                    },
                ]
            },
        }
    ]
    summary = summarize_released_owner_crossover(
        records,
        owner_discriminative_slots=["x1", "x2"],
    )
    assert summary["admission_status"] == "admitted"
    assert summary["eligible_released_owner_discriminative_edge_count"] == 1
    assert summary["crossed_to_other_endpoint_count"] == 1
    assert summary["closer_to_cued_endpoint_count"] == 0


def test_target_a_owner_crossover_is_not_admitted() -> None:
    records = [
        {
            "path": {"label": "part"},
            "arm": {"forced_coordinate_count": 1},
            "classification": {
                "slot_edge_comparisons": [
                    {
                        "slot": "x2",
                        "generation_role": "released",
                        "endpoint_separation": 219.0,
                        "closer_endpoint": "visible_whole",
                    }
                ]
            },
        }
    ]
    summary = summarize_released_owner_crossover(
        records,
        owner_discriminative_slots=[],
    )
    assert summary["admission_status"] == "not_admitted"
    assert summary["eligible_released_owner_discriminative_edge_count"] == 0
    assert summary["crossed_to_other_endpoint_count"] == 0
    assert summary["by_forced_coordinate_count"] == {}


def test_fixture_digest_verification_happens_before_payload_use(tmp_path: Path) -> None:
    payload = {
        "schema_version": "fixed_prefix_complete_box_coherence.fixture.v2",
        "targets": {},
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_verified_fixture(path, digest)["schema_version"].endswith(".v2")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_verified_fixture(path, "0" * 64)


def test_panel_path_dedup_preserves_provenance() -> None:
    target = {
        "diagnostic_paths": [
            {
                "label": "source-native",
                "box": [2, 3, 6, 7],
                "valid": True,
                "is_endpoint": True,
                "shape": {},
            }
        ],
        "panels": [
            {
                "panel": "one",
                "hybrids": [
                    {
                        "label": "same",
                        "box": [1, 2, 5, 6],
                        "valid": True,
                        "is_endpoint": True,
                        "shape": {},
                    }
                ],
            },
            {
                "panel": "two",
                "hybrids": [
                    {
                        "label": "same-again",
                        "box": [1, 2, 5, 6],
                        "valid": True,
                        "is_endpoint": False,
                        "shape": {},
                    }
                ],
            },
        ]
    }
    paths = deduplicate_panel_paths(target)
    assert len(paths) == 2
    shared = next(path for path in paths if path["box"] == [1, 2, 5, 6])
    native = next(path for path in paths if path["box"] == [2, 3, 6, 7])
    assert shared["is_endpoint"] is True
    assert {item["panel"] for item in shared["panel_provenance"]} == {"one", "two"}
    assert native["panel_provenance"] == [
        {"panel": "native_path_diagnostic", "label": "source-native"}
    ]


def test_score_record_schema_has_separate_close_score() -> None:
    vocab = 153000
    rows = []
    for selected_bin in [10, 20, 30, 40]:
        logits = torch.full((vocab,), -50.0, dtype=torch.float32)
        logits[coordinate_token_id(selected_bin)] = 2.0
        rows.append(logits)
    close = torch.full((vocab,), -50.0, dtype=torch.float32)
    close[151649] = 1.0
    result = score_coordinate_path(
        rows,
        [10, 20, 30, 40],
        box_close_logit=close,
        box_end_token_id=151649,
    )
    assert {
        "selected_coordinate_bins",
        "slot_summaries",
        "joint_primary_score_float32",
        "box_close_logprob_float32",
    } <= set(result)
    assert all(len(slot["coordinate_logits_float32"]) == 1000 for slot in result["slot_summaries"])
