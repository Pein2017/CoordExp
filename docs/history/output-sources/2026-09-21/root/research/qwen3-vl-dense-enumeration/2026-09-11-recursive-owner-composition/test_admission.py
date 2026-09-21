"""CPU-only falsification tests for the recursive admission producer."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from probes.dora_owner_learning.candidate_opportunity import digest, score
from probes.source_rweak_row_cross.run import native_record
from src.data.geometry import coord_bins_to_pixel_xyxy


ADMISSION_PATH = Path(__file__).with_name("admission.py")
TOKENIZER_PATH = Path(
    "/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
) / "tokenizer.json"


def _load_admission():
    spec = importlib.util.spec_from_file_location("recursive_owner_admission", ADMISSION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


admission = _load_admission()


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def _gt(owner, description, bbox):
    return {"object_id": str(owner), "description": description, "bbox": list(bbox)}


def _golden(gt, *, width=1000, height=1000):
    return {
        "example_id": "synthetic-example",
        "gt": gt,
        "image_height": height,
        "image_path": "/tmp/synthetic-admission-image.jpg",
        "image_width": width,
        "row_id": "synthetic-example",
        "row_index": 0,
    }


def _target(tok, owner, description, bbox):
    return admission.target_row(_gt(owner, description, bbox), tok)


def _parsed(tok, rows, gt, *, width=1000, height=1000):
    action = [token for row in rows for token in row["ids"]] + [admission.EOS]
    text = tok.decode(action, skip_special_tokens=False)
    golden = _golden(gt, width=width, height=height)
    parsed = native_record(text, {"row_id": golden["row_id"]}, golden, "im_end")
    return action, text, parsed, golden


def _record(tok, stage, action, prefix, forced, free, golden):
    text = tok.decode(action, skip_special_tokens=False)
    parsed = native_record(text, {"row_id": golden["row_id"]}, golden, "im_end")
    return {
        "example_id": golden["example_id"],
        "stage": stage,
        "action_ids": action,
        "prefix_ids": prefix,
        "forced_ids": forced,
        "free_ids": free,
        "remaining_budget": admission.CAP - len(prefix) - len(forced),
        "prefix_sha256": digest(prefix),
        "text": text,
        "stop_reason": "im_end",
        "parsed": parsed,
    }


def _composition_fixture(tok):
    """Natural source suffix and first free suffix are intentionally different."""
    old = _target(tok, "old", "person", [100, 100, 200, 200])
    first_target = _target(tok, "first", "person", [300, 100, 400, 200])
    generated_suffix = _target(tok, "generated", "person", [400, 100, 500, 200])
    second_target = _target(tok, "second", "person", [500, 100, 600, 200])
    source_suffix = _target(tok, "source-only", "unicorn", [700, 100, 800, 200])
    gt = [
        _gt("old", "person", [100, 100, 200, 200]),
        _gt("first", "person", [300, 100, 400, 200]),
        _gt("generated", "person", [400, 100, 500, 200]),
        _gt("second", "person", [500, 100, 600, 200]),
    ]
    golden = _golden(gt)
    case = {"row_id": golden["row_id"]}

    natural = old["ids"] + source_suffix["ids"] + [admission.EOS]
    first_boundary = admission.insertion(natural, first_target, tok)
    first = first_boundary["extension_ids"] + generated_suffix["ids"] + [admission.EOS]
    second_boundary = admission.insertion(
        first, second_target, tok, after=first_boundary["free_start"]
    )
    second = second_boundary["extension_ids"] + [admission.EOS]
    records = [
        _record(tok, "natural", natural, [], [], natural, golden),
        _record(
            tok,
            "first",
            first,
            first_boundary["prefix_ids"],
            first_target["ids"],
            generated_suffix["ids"] + [admission.EOS],
            golden,
        ),
        _record(
            tok,
            "second",
            second,
            second_boundary["prefix_ids"],
            second_target["ids"],
            [admission.EOS],
            golden,
        ),
    ]
    stable_score = score(
        records[0]["parsed"], seed=None, length=len(natural), stop="im_end"
    )
    frozen = {
        "example_id": golden["example_id"],
        "image_id": "synthetic-image",
        "split": "train256",
        "case": case,
        "baseline": golden,
        "stable_ids": natural,
        "stable_score": stable_score,
        "targets": [first_target, second_target],
    }
    return {
        "old": old,
        "first_target": first_target,
        "generated_suffix": generated_suffix,
        "second_target": second_target,
        "source_suffix": source_suffix,
        "golden": golden,
        "natural": natural,
        "first": first,
        "second": second,
        "first_boundary": first_boundary,
        "second_boundary": second_boundary,
        "records": records,
        "frozen": frozen,
    }


def test_insertion_uses_a_whole_row_boundary_and_exact_causal_prefix(tok):
    old = _target(tok, "old", "person", [100, 100, 200, 200])
    source = _target(tok, "source", "person", [700, 100, 800, 200])
    target = _target(tok, "target", "person", [300, 100, 400, 200])
    natural = old["ids"] + source["ids"] + [admission.EOS]

    boundary = admission.insertion(natural, target, tok)

    assert boundary["index"] == len(old["ids"])
    assert boundary["prefix_ids"] == old["ids"]
    assert boundary["extension_ids"] == old["ids"] + target["ids"]
    assert boundary["free_start"] == len(old["ids"]) + len(target["ids"])
    assert boundary["remaining_budget"] == admission.CAP - boundary["free_start"]
    assert admission.EOS not in boundary["prefix_ids"] + boundary["extension_ids"]
    assert tok.decode(boundary["prefix_ids"], skip_special_tokens=False) == old["text"]

    with pytest.raises(ValueError, match="completed first-row boundary"):
        admission.insertion(natural, target, tok, after=len(old["ids"]) + 1)


def test_consume_rejects_shifted_partition_and_bad_eos_accounting(tok):
    fixture = _composition_fixture(tok)
    natural = fixture["records"][0]
    frozen = fixture["frozen"]

    consumed = admission.consume(natural, frozen, tok)
    assert consumed["score"]["stop_reason"] == "im_end"
    assert natural["free_ids"][-1] == admission.EOS

    missing_eos = copy.deepcopy(natural)
    missing_eos["action_ids"] = missing_eos["action_ids"][:-1]
    missing_eos["free_ids"] = missing_eos["free_ids"][:-1]
    with pytest.raises(ValueError, match="action terminal corruption"):
        admission.consume(missing_eos, frozen, tok)

    duplicated_eos = copy.deepcopy(natural)
    duplicated_eos["action_ids"] = duplicated_eos["action_ids"] + [admission.EOS]
    duplicated_eos["free_ids"] = duplicated_eos["free_ids"] + [admission.EOS]
    with pytest.raises(ValueError, match="action EOS/pad corruption"):
        admission.consume(duplicated_eos, frozen, tok)

    shifted = copy.deepcopy(fixture["records"])
    first = shifted[1]
    first_boundary = fixture["first_boundary"]
    first["prefix_ids"] = first_boundary["prefix_ids"] + fixture["first_target"]["ids"][:1]
    first["forced_ids"] = fixture["first_target"]["ids"][1:]
    first["prefix_sha256"] = digest(first["prefix_ids"])
    # The concatenated history is unchanged, so only the admission boundary check can reject it.
    assert admission.consume(first, frozen, tok)["action_ids"] == fixture["first"]
    with pytest.raises(ValueError, match="first intervention boundary"):
        admission.reduce_case(shifted[:2], frozen, tok)


def test_second_insertion_retains_first_row_and_actual_generated_suffix(tok):
    fixture = _composition_fixture(tok)
    result = admission.reduce_case(fixture["records"], fixture["frozen"], tok)

    assert result["status"] == "two_correction_closure"
    second = fixture["records"][2]
    first_boundary = fixture["first_boundary"]
    assert second["prefix_ids"][: first_boundary["free_start"]] == first_boundary["extension_ids"]
    assert second["prefix_ids"][first_boundary["free_start"] :] == fixture["generated_suffix"]["ids"]
    assert second["prefix_ids"] == fixture["second_boundary"]["prefix_ids"]
    assert result["stages"][1]["forced_tokens"] == len(fixture["first_target"]["ids"])
    assert result["stages"][2]["forced_tokens"] == len(fixture["second_target"]["ids"])

    source_suffix_prefix = first_boundary["extension_ids"] + fixture["source_suffix"]["ids"]
    bad_action = source_suffix_prefix + fixture["second_target"]["ids"] + [admission.EOS]
    bad_second = _record(
        tok,
        "second",
        bad_action,
        source_suffix_prefix,
        fixture["second_target"]["ids"],
        [admission.EOS],
        fixture["golden"],
    )
    with pytest.raises(ValueError, match="second must use actual first suffix"):
        admission.reduce_case([fixture["records"][0], fixture["records"][1], bad_second], fixture["frozen"], tok)


def test_score_matches_norm1000_gt_after_pixel_conversion(tok):
    gt = [_gt("owner", "person", [100, 100, 400, 400])]
    pred = _target(tok, "pred", "person", [100, 100, 400, 400])
    action, _, parsed, _ = _parsed(tok, [pred], gt, width=640, height=480)
    card = score(parsed, seed=None, length=len(action), stop="im_end")

    assert card["50"]["tp"] == 1
    assert card["50"]["owners"] == ["owner"]
    assert card["50"]["matches"][0]["iou"] == pytest.approx(1.0)


def test_inventory_area_is_pixel_native_not_norm999_area():
    out = ADMISSION_PATH.parent
    current = json.loads((out / "candidate-inventory.json").read_text())
    rejected = json.loads((out / "candidate-inventory.luna-high-rejected.json").read_text())
    current_by_id = {str(row["example_id"]): row for row in current["candidates"]}
    rejected_by_id = {str(row["example_id"]): row for row in rejected["candidates"]}
    row = current_by_id["coco2017_train_000000184490"]
    old_row = rejected_by_id[row["example_id"]]
    owner = next(item for item in row["missing_owners"] if item["owner"] == "116077")
    old_owner = next(
        item for item in old_row["missing_owners"] if str(item["annotation_id"]) == "116077"
    )
    pixels = coord_bins_to_pixel_xyxy(
        owner["bbox_norm999"],
        image_width=row["image_width"],
        image_height=row["image_height"],
        field="synthetic.inventory_bbox",
    )
    expected = (pixels[2] - pixels[0]) * (pixels[3] - pixels[1]) / (
        row["image_width"] * row["image_height"]
    )

    assert owner["area_fraction"] == pytest.approx(expected)
    assert old_owner["area_fraction"] != pytest.approx(expected, abs=1e-6)


def test_strict_repeat_is_gt_and_class_independent_but_strictly_above_point95(tok):
    gt = [_gt("owner", "person", [0, 0, 100, 100])]
    rows = [
        _target(tok, "person", "person", [0, 0, 100, 100]),
        _target(tok, "unmatched-at-exact-point95", "unicorn", [0, 0, 100, 95]),
        _target(tok, "unmatched-above-point95", "cat", [0, 0, 100, 96]),
        _target(tok, "another-unmatched-above-point95", "dog", [0, 0, 100, 97]),
    ]
    action, _, parsed, _ = _parsed(tok, rows, gt)
    card = score(parsed, seed=None, length=len(action), stop="im_end")

    assert card["strict_repeats"] == 2
    assert card["50"]["tp"] == 1
    assert card["50"]["fp"] == 3
    assert card["invalid_predictions"] == 0
    baseline_action, _, baseline_parsed, _ = _parsed(tok, rows[:1], gt)
    baseline = score(baseline_parsed, seed=None, length=len(baseline_action), stop="im_end")
    assert "annotation_relative_FP_increase" in admission.preserving(baseline, card, "im_end")
