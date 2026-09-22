"""Focused CPU invariants for the positive-branch input package."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "build_input_package.py"
MANIFEST_PATH = ROOT / "candidate_manifest.json"


def load_builder():
    spec = importlib.util.spec_from_file_location("positive_branch_input_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return load_builder()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_is_source_bound_and_cpu_only(builder, manifest):
    builder.validate_manifest(manifest, check_sources=True)
    assert manifest["status"] == "candidate_only"
    assert manifest["immutable_candidate_manifest"] is True
    assert manifest["counts"] == {
        "positive_count": 3,
        "normal_count": 56,
        "positive_c_tokens": 30,
        "positive_w_tokens": 29,
        "positive_h_tokens": 161,
        "normal_action_tokens": 6056,
        "normal_kl_positions": 6047,
    }
    assert manifest["build_receipt"]["no_model_import"] is True
    assert manifest["build_receipt"]["no_tokenizer_import"] is True
    assert manifest["build_receipt"]["no_gpu_calls"] is True


def test_rebuild_reproduces_literals_and_fresh_source_slices(builder, manifest):
    # Rebuild from the immutable source files and compare the decision-bearing
    # positive payload.  This catches accidental hand-entered or retokenized
    # h/c/w literals while ignoring only the outer filesystem receipt.
    rebuilt = builder.build_manifest()
    assert rebuilt["positives"] == manifest["positives"]
    for case in manifest["positives"]:
        h = case["h"]["token_ids"]
        c = case["c"]["token_ids"]
        w = case["w"]["token_ids"]
        offsets = case["continuation_offsets"]
        assert offsets["c_target_span"] == [len(h), len(h) + len(c)]
        assert offsets["w_only_kl_span"] == [len(h) + len(c), len(h) + len(c) + len(w)]
        assert case["w"]["coord_bins"] == builder.EXPECTED_W[case["candidate_id"]]["coord_bins"]
        assert case["w"]["description"] == builder.EXPECTED_W[case["candidate_id"]]["description"]


def test_shifted_c_boundary_is_rejected(builder, manifest):
    mutated = copy.deepcopy(manifest)
    mutated["positives"][0]["continuation_offsets"]["c_target_span"][0] += 1
    with pytest.raises(ValueError, match="shifted c target boundary"):
        builder.validate_manifest(mutated)


@pytest.mark.parametrize("mutation,expected", [
    ("extra_w_eos", "w literal/hash invalid|extra EOS"),
    ("eos_in_w_mask", "w positions do not cover w exactly|EOS is silently supervised"),
    ("missing_source_eos", "source EOS evidence absent"),
])
def test_eos_cannot_be_silently_supervised(builder, manifest, mutation, expected):
    mutated = copy.deepcopy(manifest)
    case = mutated["positives"][0]
    if mutation == "extra_w_eos":
        case["w"]["token_ids"].append(builder.IM_END)
    elif mutation == "eos_in_w_mask":
        case["continuation_offsets"]["w_only_kl_positions"].append(
            case["continuation_offsets"]["terminal_eos_position_in_source_action"]
        )
    else:
        case["source_record"]["native_eos_observed"] = False
    with pytest.raises(ValueError, match=expected):
        builder.validate_manifest(mutated)


def test_normal_mask_preserves_known_geometry_invalid_row(builder, manifest):
    normal = next(case for case in manifest["normals"]["cases"] if case["image_id"] == "360573")
    layout = normal["initial_layout"]
    assert layout["invalid_geometry_rows"] == [8]
    assert layout["parser_drops"] == 1
    drop = layout["parser_drop_rows"][0]
    assert drop["generated_order"] == 8
    assert len(drop["token_positions"]) == 9
    assert set(drop["token_positions"]).isdisjoint(layout["kl_positions"])
    assert manifest["normals"]["mask_summary"]["parser_nonclean_reference_count"] == 1
    assert manifest["normals"]["mask_summary"]["geometry_invalid_excluded_positions"]["360573"] == drop["token_positions"]


def test_positive_normal_image_sets_are_disjoint(builder, manifest):
    positive = {str(case["image"]["image_id"]) for case in manifest["positives"]}
    normal = {str(case["image_id"]) for case in manifest["normals"]["cases"]}
    assert positive == {"351017", "417044", "477415"}
    assert len(normal) == 56
    assert positive.isdisjoint(normal)
