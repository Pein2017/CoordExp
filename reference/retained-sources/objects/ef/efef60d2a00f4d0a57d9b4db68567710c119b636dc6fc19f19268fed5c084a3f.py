"""Contract tests for the emitted native-escape candidate manifest.

These tests exercise the same literal h+c consumer used by the future runner;
they do not call a model and never use GT fields.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "build_candidate_manifest.py"
MANIFEST_PATH = ROOT / "candidate_manifest.json"


def load_builder():
    spec = importlib.util.spec_from_file_location("native_escape_manifest_builder", BUILDER_PATH)
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


def _case(manifest, case_id):
    return next(case for case in manifest["cases"] if case["case_id"] == case_id)


def _source_rows(builder, case, record, tokenizer, census):
    rows = builder.parse_rows(record["action_ids"], record["_packet_case"], tokenizer, census)
    prefix_length = len(record["extension_ids"])
    return [row for row in rows if row["token_start"] >= prefix_length]


def _source_context(builder, manifest, case_id):
    case = _case(manifest, case_id)
    packet = json.loads(Path(manifest["source_packet"]["path"]).read_text(encoding="utf-8"))
    packet_case = next(value for value in packet["cases"] if value["case_id"] == case_id)
    path = Path(case["source_translated_record"]["path"])
    record = next(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("case_id") == case_id
        and json.loads(line).get("job_id") == builder.SOURCE_JOB_ID
    )
    record_case = dict(packet_case)
    record["_packet_case"] = record_case
    return case, record, builder.load_tokenizer(), builder.load_source_census()


def test_emitted_manifest_and_real_h_c_consumer(builder, manifest):
    builder.validate_manifest(manifest)
    assert manifest["status"] == "candidate_only"
    assert manifest["admission"].startswith("not_visually_admitted")
    assert manifest["frozen_cohort"] == list(builder.COHORT)
    for case_id in builder.COHORT:
        case = _case(manifest, case_id)
        assert case["h"]["job_id"] == builder.H_JOB_ID
        assert case["source_translated_record"]["job_id"] == builder.SOURCE_JOB_ID
        for candidate in case["candidates"]:
            consumed = builder.consume_candidate(manifest, case_id, candidate["candidate_id"])
            assert consumed["action_ids"] == consumed["h_ids"] + consumed["c_ids"]
            assert len(consumed["c_ids"]) == candidate["c_length"]
            assert consumed["c_text"] == candidate["c_text"]


def test_h_identity_is_exact_packet_early_original_native(builder, manifest):
    packet = json.loads(Path(manifest["source_packet"]["path"]).read_text(encoding="utf-8"))
    for case_id in builder.COHORT:
        packet_case = next(value for value in packet["cases"] if value["case_id"] == case_id)
        expected_h = next(job for job in packet_case["jobs"] if job["job_id"] == builder.H_JOB_ID)["extension_ids"]
        emitted_h = _case(manifest, case_id)["h"]
        assert emitted_h["ids"] == expected_h
        assert emitted_h["length"] == len(expected_h)
        assert emitted_h["ids_sha256"] == builder.sha256_ids(expected_h)
        assert emitted_h["text_sha256"] == builder.sha256_text(emitted_h["text"])


def test_literal_c_sequence_and_source_slice(builder, manifest):
    for case_id in builder.COHORT:
        case, record, tokenizer, census = _source_context(builder, manifest, case_id)
        free_rows = _source_rows(builder, case, record, tokenizer, census)
        complete_rows = [row for row in free_rows if row["complete_canonical_row"]]
        by_order = {row["raw_row_index"]: row for row in complete_rows}
        for candidate in case["candidates"]:
            source_row = by_order[candidate["source_generated_order"]]
            assert candidate["source_job_id"] == builder.SOURCE_JOB_ID
            assert candidate["c_ids"] == source_row["token_ids"]
            assert candidate["c_token_texts"] == source_row["token_texts"]
            assert candidate["c_text"] == source_row["raw_text"]
            assert candidate["c_length"] == len(candidate["c_ids"])
            assert candidate["c_ids_sha256"] == builder.sha256_ids(candidate["c_ids"])
            assert candidate["c_text_sha256"] == builder.sha256_text(candidate["c_text"])
            assert tokenizer.decode(candidate["c_ids"], skip_special_tokens=False) == candidate["c_text"]


def test_parser_invalid_and_incomplete_burden_is_preserved(builder, manifest):
    for case_id in builder.COHORT:
        case, record, tokenizer, census = _source_context(builder, manifest, case_id)
        free_rows = _source_rows(builder, case, record, tokenizer, census)
        expected_invalid = {
            (row["raw_row_index"], row["native_status"], row["raw_text_sha256"])
            for row in free_rows
            if not row["geometry_valid"]
        }
        expected_incomplete = {
            (row["raw_row_index"], row["native_status"], row["raw_text_sha256"])
            for row in free_rows
            if not row["complete_canonical_row"]
        }
        source_free = _case(manifest, case_id)["source_free"]
        actual_invalid = {
            (row["raw_row_index"], row["native_status"], row["raw_text_sha256"])
            for row in source_free["raw_invalid_rows"]
        }
        actual_incomplete = {
            (row["raw_row_index"], row["native_status"], row["raw_text_sha256"])
            for row in source_free["raw_incomplete_rows"]
        }
        assert actual_invalid == expected_invalid
        assert actual_incomplete == expected_incomplete
        assert source_free["invalid_row_count"] == len(expected_invalid)
        assert source_free["incomplete_row_count"] == len(expected_incomplete)
        for row in source_free["raw_invalid_rows"] + source_free["raw_incomplete_rows"]:
            assert row["token_text_reconstructs_raw"] is True
            assert row["token_ids_sha256"] == builder.sha256_ids(row["token_ids"])


def test_first_five_selection_order_and_no_backfill(builder, manifest):
    expected_orders = {"351017": [2, 3], "417044": [8, 9], "477415": [7, 8], "502725": [5]}
    for case_id, expected in expected_orders.items():
        case = _case(manifest, case_id)
        first_five = case["source_free"]["first_five_complete_rows"]
        assert [row["raw_row_index"] for row in first_five] == sorted(row["raw_row_index"] for row in first_five)
        assert [candidate["source_generated_order"] for candidate in case["candidates"]] == expected
        assert case["selection_counts"]["selected"] == len(expected)
        assert case["selection_counts"]["held"] == len(first_five) - len(expected)
        assert len(case["candidates"]) <= builder.MAX_CANDIDATES
    assert manifest["counts"]["source_shortfall_case_ids"] == ["502725"]


def test_class_blind_iou_against_every_h_and_prior_candidate(builder, manifest):
    for case_id in builder.COHORT:
        case = _case(manifest, case_id)
        h_valid = [row for row in case["h"]["rows"] if row["geometry_valid"]]
        chosen = []
        for candidate in case["candidates"]:
            box = candidate["bbox_pixel_xyxy"]
            h_scores = [builder.iou_xyxy(box, row["bbox_pixel_xyxy"]) for row in h_valid]
            prior_scores = [builder.iou_xyxy(box, row["bbox_pixel_xyxy"]) for row in chosen]
            assert max(h_scores, default=0.0) == pytest.approx(candidate["max_iou_to_h"])
            assert max(prior_scores, default=0.0) == pytest.approx(candidate["max_iou_to_prior_candidates"])
            assert all(score <= builder.IOU_LIMIT for score in h_scores)
            assert all(score <= builder.IOU_LIMIT for score in prior_scores)
            chosen.append(candidate)


def test_source_is_only_early_original_translated(builder, manifest):
    for case_id in builder.COHORT:
        case = _case(manifest, case_id)
        assert case["source_translated_record"]["boundary"] == "early"
        assert case["source_translated_record"]["image_condition"] == "original"
        assert case["source_translated_record"]["history_condition"] == "translated"
        for candidate in case["candidates"]:
            assert candidate["source_job_id"] == builder.SOURCE_JOB_ID
            mutated = copy.deepcopy(manifest)
            mutated_case = _case(mutated, case_id)
            mutated_candidate = next(
                value for value in mutated_case["candidates"] if value["candidate_id"] == candidate["candidate_id"]
            )
            mutated_candidate["source_job_id"] = builder.H_JOB_ID
            with pytest.raises(ValueError, match="early_original_translated"):
                builder.consume_candidate(mutated, case_id, candidate["candidate_id"])


def test_candidate_figures_exist_and_are_geometry_only(builder, manifest):
    for case in manifest["cases"]:
        figure_by_candidate = {figure["candidate_id"]: figure for figure in case["figures"]}
        for candidate in case["candidates"]:
            figure = figure_by_candidate[candidate["candidate_id"]]
            assert Path(figure["full_path"]).is_file()
            assert Path(figure["crop_path"]).is_file()
            assert figure["annotation"] == "candidate geometry only; no GT or classification annotation"
            assert builder.sha256_file(Path(figure["full_path"])) == figure["full_sha256"]
            assert builder.sha256_file(Path(figure["crop_path"])) == figure["crop_sha256"]
