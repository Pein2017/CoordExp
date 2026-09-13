from collections import Counter
import json

import pytest

from probes.native_owner_scale import scale


def _row(example_id, *, overlap=0, fn=0, recall=1.0, cap=0, drops=0):
    return {"example_id": example_id, "stable_overlap_counts": {"80": overlap},
            "stable_score": {"cap": cap, "parser_drops": drops,
                             "50": {"fn": fn, "recall": recall}},
            "golden": {"gt": [{}]}}


def test_strata_keep_repeat_missing_and_normal_separate():
    assert scale.stable_stratum(_row("r", overlap=1, fn=0)) == "repeat_or_drift"
    assert scale.stable_stratum(_row("m", fn=2, recall=0.5)) == "missing_or_early_stop"
    assert scale.stable_stratum(_row("n", fn=1, recall=0.8)) == "normal_single_miss"
    assert scale.stable_stratum(_row("f")) == "normal_control"


@pytest.mark.parametrize("n", [16, 17, 31, 32])
def test_batch2_schedule_has_16n_steps_and_32_equal_exposures(n):
    ids = [f"p{i}" for i in range(n)]
    steps = scale.exposure_steps(ids)
    assert len(steps) == 16 * n
    assert {len(step) for step in steps} == {2}
    assert Counter(x["record_id"] for step in steps for x in step) == Counter({x: 32 for x in ids})


def test_local_w_requires_immediate_literal_nonstrict_duplicate():
    text = ("<|object_ref_start|>cat<|object_ref_end|><|box_start|>"
            "<|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>")
    ids = [151646, 10, 151647, 151648, 151671, 151672, 151673, 151674, 151649]

    class Tokenizer:
        def encode(self, value, add_special_tokens=False):
            assert value == text and not add_special_tokens
            return ids

    parsed = {"pred": [{"generated_order": 0, "raw_span_text": text,
                         "bbox": [10, 10, 20, 20], "description": "cat"}]}
    value = scale._literal_first_w(ids + [151645], text + "<|im_end|>", parsed,
                                   [[100, 100, 110, 110]], Tokenizer())
    assert value["status"] == "candidate_local_w"
    duplicate = scale._literal_first_w(ids, text, parsed, [[10, 10, 20, 20]], Tokenizer())
    assert duplicate["status"] == "strict_duplicate_local_w"


def test_local_w_does_not_skip_malformed_or_repeated_leading_content():
    class Tokenizer:
        def encode(self, value, add_special_tokens=False):
            return [151646, 10, 151647, 151648, 151671, 151672, 151673, 151674, 151649]

    text = "<|object_ref_start|>cat<|object_ref_end|><|box_start|><|coord_1|><|coord_2|><|coord_3|><|coord_4|><|box_end|>"
    parsed = {"pred": [{"generated_order": 1, "raw_span_text": text,
                         "bbox": [10, 10, 20, 20], "description": "cat"}]}
    assert scale._literal_first_w([], "junk" + text, parsed, [], Tokenizer())["status"] == \
        "first_free_content_not_one_literal_complete_row"


def test_native_history_uses_maximal_complete_prefix_without_repair():
    row = [151646, 1, 2, 3, 4, 5, 6, 151649]
    prefix, tail = scale._maximal_complete_prefix(row + [151646, 9, 10])
    assert prefix == row
    assert tail == [151646, 9, 10]


def test_adapter_identity_allows_only_known_metadata_rename():
    stable = {"kind": "dora_adapter", "root": "/a", "file_count": 1,
              "files": [{"relative_path": "x", "sha256": "abc", "size_bytes": 3}],
              "semantic_identity": {"r": 16}, "tensor_manifest": {"target_count": 1},
              "version": "coordexp-swift-dora-adapter-v1", "fingerprint": "old-derived"}
    renamed = {**stable, "version": "coordexp-infras-dora-adapter-v1",
               "fingerprint": "new-derived"}
    assert scale.same_adapter_payload(stable, renamed)
    assert not scale.same_adapter_payload(stable, {**renamed, "files": []})
    assert not scale.same_adapter_payload(stable, {**renamed, "tensor_manifest": {"target_count": 2}})
    assert not scale.same_adapter_payload(stable, {**renamed, "version": "unknown"})
    assert not scale.same_adapter_payload(renamed, stable)


def test_selection_source_bridge_accepts_only_exact_preserved_producer(tmp_path, monkeypatch):
    snapshot = tmp_path / "executed.py"
    snapshot.write_text("executed producer")
    monkeypatch.setattr(scale, "EXECUTED_PRODUCER_SNAPSHOT", snapshot)
    source = {"path": scale.__file__, "sha256": scale.file_hash(snapshot)}
    assert scale.valid_selection_source("producer", source)
    assert not scale.valid_selection_source("stable50_universe", source)
    snapshot.write_text("changed snapshot")
    assert not scale.valid_selection_source("producer", source)


def test_remainder_is_exact_disjoint_complement():
    full = [f"j{i}" for i in range(156)]
    selected = full[:4]
    remainder = [job for job in full if job not in selected]
    assert len(selected) == 4 and len(remainder) == 152
    assert set(selected).isdisjoint(remainder)
    assert set(selected) | set(remainder) == set(full)


def test_bank_admission_prefers_distinct_images_before_duplicate_packages():
    rows = ([{"job_id": f"first-{i}", "example_id": f"image-{i}",
              "admission_priority": i} for i in range(28)]
            + [{"job_id": f"duplicate-{i}", "example_id": f"image-{i}",
                "admission_priority": 28 + i} for i in range(4)]
            + [{"job_id": f"late-first-{i}", "example_id": f"image-{28 + i}",
                "admission_priority": 32 + i} for i in range(4)])
    assert len({row["example_id"] for row in rows[:32]}) == 28
    admitted = scale.distinct_first_admission(rows, maximum=32)
    assert len(admitted) == 32
    assert len({row["example_id"] for row in admitted}) == 32
    assert [row["admission_priority"] for row in admitted[-4:]] == [32, 33, 34, 35]


def test_final_reviews_are_bound_to_exact_acquisition(tmp_path):
    result = tmp_path / "acquisition.json"
    result.write_text(json.dumps({"rows": []}))
    reviews = {
        "schema": "native_owner_scale.visual_reviews.final.v1",
        "status": "lead_accepted_physical_bank_admission",
        "acquisition_result": scale.binding(result),
        "counts": {"accept": 16, "neutral": 140},
        "candidate_decisions": {"accept": 16, "neutral": 34},
    }
    assert scale.validate_final_reviews(reviews, result) is reviews
    result.write_text(json.dumps({"rows": ["changed"]}))
    with pytest.raises(ValueError, match="visual review/acquisition binding"):
        scale.validate_final_reviews(reviews, result)
