#!/usr/bin/env python3
"""Post-hoc concentration diagnostic for the verified endpoint-A result.

This reducer only projects frozen Stable50 fields and the verified A consumer.
The full 384/256/128 panels remain primary; exclusions are descriptive only.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = ROOT / "endpoint-A"
PACKET = ROOT / "endpoint-preparation" / "packet.json"
CONSUMER = ENDPOINT / "consumer.json"
REDUCTION = ENDPOINT / "reduction.json"
OUTPUT = ENDPOINT / "concentration-diagnostic.json"
THRESHOLDS = ("50", "60", "80")
FOCUS_IMAGES = (351017, 417044, 477415, 502725, 39654)
ORIGINAL_TRAIN_CAPS = (351017, 417044, 477415, 502725)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def drop_taxonomy(parsed: dict[str, Any]) -> dict[str, Any]:
    dropped = parsed["dropped_predictions"]
    reasons = Counter(str(row.get("reason", "missing_reason")) for row in dropped)
    geometry = sum(row.get("reason") == "geometry_invalid" for row in dropped)
    malformed = len(dropped) - geometry
    complete = sum(str(row.get("raw_span_text", "")).endswith("<|box_end|>") for row in dropped)
    geometry_complete = sum(
        row.get("reason") == "geometry_invalid"
        and str(row.get("raw_span_text", "")).endswith("<|box_end|>")
        for row in dropped)
    malformed_complete = sum(
        row.get("reason") != "geometry_invalid"
        and str(row.get("raw_span_text", "")).endswith("<|box_end|>")
        for row in dropped)
    return {
        "total": len(dropped),
        "geometry_invalid": geometry,
        "malformed_or_incomplete": malformed,
        "geometry_invalid_complete": geometry_complete,
        "geometry_invalid_incomplete": geometry - geometry_complete,
        "other_complete_malformed": malformed_complete,
        "other_incomplete_malformed": malformed - malformed_complete,
        "complete_rows": complete,
        "incomplete_fragments": len(dropped) - complete,
        "reason_counts": dict(sorted(reasons.items())),
    }


def burden(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    if side == "anchor":
        parsed_rows = [row["stable_parsed"] for row in rows]
        scores = [row["stable_score"] for row in rows]
        texts = [row["stable_parsed"]["raw_decode_text"] for row in rows]
    else:
        parsed_rows = [row["candidate"]["parsed"] for row in rows]
        scores = [row["candidate"]["score"] for row in rows]
        texts = [row["candidate"]["text"] for row in rows]
    taxonomies = [drop_taxonomy(parsed) for parsed in parsed_rows]
    return {
        "raw_object_starts": sum(text.count("<|object_ref_start|>") for text in texts),
        "parsed_rows": sum(len(parsed["pred"]) for parsed in parsed_rows),
        "dropped_rows": sum(item["total"] for item in taxonomies),
        "dropped_geometry_invalid": sum(item["geometry_invalid"] for item in taxonomies),
        "dropped_malformed_or_incomplete": sum(item["malformed_or_incomplete"] for item in taxonomies),
        "dropped_geometry_invalid_complete": sum(item["geometry_invalid_complete"] for item in taxonomies),
        "dropped_geometry_invalid_incomplete": sum(item["geometry_invalid_incomplete"] for item in taxonomies),
        "dropped_other_complete_malformed": sum(item["other_complete_malformed"] for item in taxonomies),
        "dropped_other_incomplete_malformed": sum(item["other_incomplete_malformed"] for item in taxonomies),
        "dropped_complete_rows": sum(item["complete_rows"] for item in taxonomies),
        "dropped_incomplete_fragments": sum(item["incomplete_fragments"] for item in taxonomies),
        "drop_reason_counts": dict(sorted(sum((Counter(item["reason_counts"]) for item in taxonomies), Counter()).items())),
        "strict_repeats": sum(score["strict_repeats"] for score in scores),
        "complete_token_length": sum(score["complete_token_length"] for score in scores),
        "caps": sum(score["cap"] for score in scores),
        "cap_image_ids": sorted(int(row["image_id"]) for row, score in zip(rows, scores) if score["cap"]),
    }


def panel(rows: list[dict[str, Any]], label: str, posthoc: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": label,
        "images": len(rows),
        "posthoc_descriptive_only": posthoc,
        "anchor": {},
        "candidate": {},
        "owner_changes": {},
    }
    for threshold in THRESHOLDS:
        for side in ("anchor", "candidate"):
            scores = [row["stable_score"] if side == "anchor" else row["candidate"]["score"] for row in rows]
            tp = sum(score[threshold]["tp"] for score in scores)
            fp = sum(score[threshold]["fp"] for score in scores)
            fn = sum(score[threshold]["fn"] for score in scores)
            result[side][threshold] = {"tp": tp, "fp": fp, "fn": fn, "f1": f1(tp, fp, fn)}
        changes = []
        for row in rows:
            before = set(row["stable_score"][threshold]["owners"])
            after = set(row["candidate"]["score"][threshold]["owners"])
            changes.append((before, after))
        result["owner_changes"][threshold] = {
            "gained": sum(len(after - before) for before, after in changes),
            "lost": sum(len(before - after) for before, after in changes),
            "retained": sum(len(before & after) for before, after in changes),
        }
    result["anchor"]["burden"] = burden(rows, "anchor")
    result["candidate"]["burden"] = burden(rows, "candidate")
    return result


def iou(left: list[int], right: list[int]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def token_span(tokenizer: Tokenizer, ids: list[int], text: str, row: dict[str, Any]) -> dict[str, Any]:
    start = len(tokenizer.encode(text[: row["char_start"]], add_special_tokens=False).ids)
    end = len(tokenizer.encode(text[: row["char_end"]], add_special_tokens=False).ids)
    assert tokenizer.decode(ids[:start], skip_special_tokens=False) == text[: row["char_start"]]
    assert tokenizer.decode(ids[:end], skip_special_tokens=False) == text[: row["char_end"]]
    return {
        "action_token_start_inclusive": start,
        "action_token_end_exclusive": end,
        "char_start_inclusive": row["char_start"],
        "char_end_exclusive": row["char_end"],
        "literal_prefix_token_count": 0,
    }


def first_repeat(tokenizer: Tokenizer, ids: list[int], text: str, parsed: dict[str, Any], expected: int) -> dict[str, Any] | None:
    valid = sorted(parsed["pred"], key=lambda row: row["generated_order"])
    repeat_count = 0
    first: dict[str, Any] | None = None
    for current_index, current in enumerate(valid):
        earlier = [(iou(current["bbox"], prior["bbox"]), prior_index, prior)
                   for prior_index, prior in enumerate(valid[:current_index])]
        matches = [item for item in earlier if item[0] > 0.95]
        if not matches:
            continue
        repeat_count += 1
        if first is not None:
            continue
        overlap, prior_index, prior = max(matches, key=lambda item: (item[0], -item[1]))
        first = {
            "threshold": "class_blind_pixel_iou_gt_0.95",
            "iou": overlap,
            "current_pred_index": current_index,
            "current_generated_order": current["generated_order"],
            "current_description": current["description"],
            "current_bbox_pixel_xyxy": current["bbox"],
            "current_raw_span_sha256": current["raw_span_sha256"],
            "current_offsets": token_span(tokenizer, ids, text, current),
            "prior_pred_index": prior_index,
            "prior_generated_order": prior["generated_order"],
            "prior_description": prior["description"],
            "prior_bbox_pixel_xyxy": prior["bbox"],
            "prior_raw_span_sha256": prior["raw_span_sha256"],
            "prior_offsets": token_span(tokenizer, ids, text, prior),
        }
    assert repeat_count == expected, (repeat_count, expected)
    return first


def image_summary(tokenizer: Tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    anchor_parsed = row["stable_parsed"]
    candidate = row["candidate"]
    candidate_parsed = candidate["parsed"]
    summary: dict[str, Any] = {
        "image_id": int(row["image_id"]),
        "example_id": row["example_id"],
        "split": row["split"],
        "scores": {},
        "owner_changes": {},
        "anchor": {
            "raw_object_starts": anchor_parsed["raw_decode_text"].count("<|object_ref_start|>"),
            "parsed_rows": len(anchor_parsed["pred"]),
            "dropped": drop_taxonomy(anchor_parsed),
            "strict_repeats": row["stable_score"]["strict_repeats"],
            "cap": row["stable_score"]["cap"],
            "stop_reason": row["stable_score"]["stop_reason"],
            "action_token_count": len(row["stable_ids"]),
        },
        "candidate": {
            "raw_object_starts": candidate["text"].count("<|object_ref_start|>"),
            "parsed_rows": len(candidate_parsed["pred"]),
            "dropped": drop_taxonomy(candidate_parsed),
            "strict_repeats": candidate["score"]["strict_repeats"],
            "cap": candidate["score"]["cap"],
            "stop_reason": candidate["score"]["stop_reason"],
            "action_token_count": len(candidate["action_ids"]),
        },
    }
    for threshold in THRESHOLDS:
        a = row["stable_score"][threshold]
        b = candidate["score"][threshold]
        summary["scores"][threshold] = {
            "anchor": {key: a[key] for key in ("tp", "fp", "fn", "f1")},
            "candidate": {key: b[key] for key in ("tp", "fp", "fn", "f1")},
            "delta": {key: b[key] - a[key] for key in ("tp", "fp", "fn", "f1")},
        }
        before, after = set(a["owners"]), set(b["owners"])
        summary["owner_changes"][threshold] = {
            "gained": sorted(after - before),
            "lost": sorted(before - after),
            "retained": sorted(before & after),
        }
    summary["anchor"]["first_strict_repeat"] = first_repeat(
        tokenizer, row["stable_ids"], anchor_parsed["raw_decode_text"], anchor_parsed,
        row["stable_score"]["strict_repeats"])
    summary["candidate"]["first_strict_repeat"] = first_repeat(
        tokenizer, candidate["action_ids"], candidate["text"], candidate_parsed,
        candidate["score"]["strict_repeats"])
    return summary


def ranking_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compact = []
    for row in rows:
        item = {
            "image_id": int(row["image_id"]),
            "additional_repeats": row["candidate"]["score"]["strict_repeats"] - row["stable_score"]["strict_repeats"],
            "additional_drops": row["candidate"]["score"]["parser_drops"] - row["stable_score"]["parser_drops"],
            "additional_fp": {},
            "owner_losses": {},
            "owner_gains": {},
        }
        for threshold in THRESHOLDS:
            item["additional_fp"][threshold] = row["candidate"]["score"][threshold]["fp"] - row["stable_score"][threshold]["fp"]
            before = set(row["stable_score"][threshold]["owners"])
            after = set(row["candidate"]["score"][threshold]["owners"])
            item["owner_losses"][threshold] = len(before - after)
            item["owner_gains"][threshold] = len(after - before)
        compact.append(item)
    return {
        "additional_fp": {threshold: sorted(
            ({"image_id": item["image_id"], "value": item["additional_fp"][threshold]} for item in compact),
            key=lambda item: (-item["value"], item["image_id"])) for threshold in THRESHOLDS},
        "additional_repeats": sorted(
            ({"image_id": item["image_id"], "value": item["additional_repeats"]} for item in compact),
            key=lambda item: (-item["value"], item["image_id"])),
        "additional_drops": sorted(
            ({"image_id": item["image_id"], "value": item["additional_drops"]} for item in compact),
            key=lambda item: (-item["value"], item["image_id"])),
        "owner_losses": {threshold: sorted(
            ({"image_id": item["image_id"], "lost": item["owner_losses"][threshold],
              "gained": item["owner_gains"][threshold]} for item in compact),
            key=lambda item: (-item["lost"], item["image_id"])) for threshold in THRESHOLDS},
    }


def concentration(rows: list[dict[str, Any]], image_id: int) -> dict[str, Any]:
    focus = next(row for row in rows if int(row["image_id"]) == image_id)
    result: dict[str, Any] = {"image_id": image_id, "metrics": {}}
    metrics = {
        "additional_repeats": lambda row: row["candidate"]["score"]["strict_repeats"] - row["stable_score"]["strict_repeats"],
        "additional_drops": lambda row: row["candidate"]["score"]["parser_drops"] - row["stable_score"]["parser_drops"],
    }
    for threshold in THRESHOLDS:
        metrics[f"additional_fp_{threshold}"] = lambda row, t=threshold: row["candidate"]["score"][t]["fp"] - row["stable_score"][t]["fp"]
        metrics[f"owner_losses_{threshold}"] = lambda row, t=threshold: len(set(row["stable_score"][t]["owners"]) - set(row["candidate"]["score"][t]["owners"]))
    for name, getter in metrics.items():
        values = [getter(row) for row in rows]
        value = getter(focus)
        positive_total = sum(max(0, item) for item in values)
        result["metrics"][name] = {
            "value": value,
            "dev_net_total": sum(values),
            "share_of_dev_net_change": value / sum(values) if sum(values) else None,
            "dev_positive_worsening_total": positive_total,
            "share_of_positive_worsening": value / positive_total if value > 0 and positive_total else 0.0,
            "rank_descending": 1 + sum(item > value for item in values),
        }
    return result


def main() -> None:
    packet = load(PACKET)
    consumer = load(CONSUMER)
    frozen_reduction = load(REDUCTION)
    assert len(consumer) == 384
    packet_by_id = {row["example_id"]: row for row in packet["eval_records"]}
    assert set(packet_by_id) == {row["example_id"] for row in consumer}
    rows = []
    for candidate in consumer:
        stable = packet_by_id[candidate["example_id"]]
        assert candidate["score"] == next(row for row in frozen_reduction["per_image"]
                                           if row["example_id"] == candidate["example_id"])["candidate"]
        rows.append({**stable, "candidate": candidate})

    split_selectors = {
        "online8": lambda row: row["split"] == "online8",
        "reference56": lambda row: row["split"] == "reference56",
        "remaining192": lambda row: row["split"] == "remaining192",
        "train256": lambda row: row["split"] != "dev128",
        "dev128": lambda row: row["split"] == "dev128",
        "union384": lambda row: True,
    }
    panels = {label: panel([row for row in rows if selector(row)], label)
              for label, selector in split_selectors.items()}
    # Exact parity with the frozen endpoint reducer for decision metrics.
    for label, frozen in frozen_reduction["panels"].items():
        projected = panels[label]
        for side in ("anchor", "candidate"):
            for threshold in THRESHOLDS:
                assert projected[side][threshold] == frozen[side][threshold]
        assert projected["owner_changes"] == frozen["owner_changes"]

    dev = [row for row in rows if row["split"] == "dev128"]
    train = [row for row in rows if row["split"] != "dev128"]
    posthoc = {
        "dev127_excluding_new_cap_39654": panel(
            [row for row in dev if int(row["image_id"]) != 39654],
            "dev127_excluding_new_cap_39654", True),
        "train252_excluding_original_four_caps": panel(
            [row for row in train if int(row["image_id"]) not in ORIGINAL_TRAIN_CAPS],
            "train252_excluding_original_four_caps", True),
    }
    tokenizer = Tokenizer.from_file(str(Path(packet["model"]["base_model_path"]) / "tokenizer.json"))
    focus = {str(image_id): image_summary(tokenizer, next(row for row in rows if int(row["image_id"]) == image_id))
             for image_id in FOCUS_IMAGES}
    dev127 = posthoc["dev127_excluding_new_cap_39654"]
    train252 = posthoc["train252_excluding_original_four_caps"]
    decision = {
        "observations": [
            "Image 39654 is the sole new cap and dominates dev additional parser drops, strict repeats, and false positives.",
            "Excluding 39654 removes the catastrophic cap but leaves lower candidate F1 and more owner losses than gains at all thresholds on dev127.",
            "Excluding the four original Stable50 train caps reverses the apparent train256 improvement: candidate F1 is lower at all thresholds on train252, with more owner losses than gains.",
        ],
        "inference": "The endpoint is not explained by one isolated loop migration alone: cap/repetition burden is highly concentrated, while annotation-relative preservation damage is diffuse across non-cap dev and train images.",
        "next_bounded_probe": "Separate the mechanisms: inspect the saved first-repeat onset for 39654 against resolved old-cap onsets, and visually adjudicate the highest-ranked non-cap owner-loss/FP dev cases before considering any new training or aggregate endpoint.",
        "dev127_candidate_minus_anchor_f1": {
            threshold: dev127["candidate"][threshold]["f1"] - dev127["anchor"][threshold]["f1"]
            for threshold in THRESHOLDS
        },
        "dev127_owner_gain_minus_loss": {
            threshold: dev127["owner_changes"][threshold]["gained"] - dev127["owner_changes"][threshold]["lost"]
            for threshold in THRESHOLDS
        },
        "train252_candidate_minus_anchor_f1": {
            threshold: train252["candidate"][threshold]["f1"] - train252["anchor"][threshold]["f1"]
            for threshold in THRESHOLDS
        },
        "train252_owner_gain_minus_loss": {
            threshold: train252["owner_changes"][threshold]["gained"] - train252["owner_changes"][threshold]["lost"]
            for threshold in THRESHOLDS
        },
    }

    output = {
        "schema": "positive_branch_vs_repeat_event.endpoint_A_concentration_diagnostic.v1",
        "status": "complete",
        "claim_boundary": {
            "primary_metrics": "The original union384/train256/dev128 metrics remain decision-owning.",
            "posthoc": "The dev127 and train252 exclusions are descriptive concentration checks only, not acceptance metrics.",
            "semantics": "Annotation-relative owner and parser/geometry evidence only; no physical-owner, hallucination, or visual-quality assertion.",
            "invalid_note": "score.invalid_predictions is zero because geometry-invalid and malformed rows are parser drops; actual dropped-row taxonomy is reported separately.",
        },
        "inputs": {
            "packet": {"path": str(PACKET), "sha256": sha256(PACKET)},
            "consumer": {"path": str(CONSUMER), "sha256": sha256(CONSUMER)},
            "reduction": {"path": str(REDUCTION), "sha256": sha256(REDUCTION)},
            "reducer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "primary_panels": panels,
        "posthoc_exclusions": posthoc,
        "dev_image_rankings": ranking_rows(dev),
        "new_cap_39654_concentration": concentration(dev, 39654),
        "focus_images": focus,
        "decision": decision,
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    assert load(OUTPUT) == output
    print(json.dumps({"status": "complete", "output": str(OUTPUT), "sha256": sha256(OUTPUT)}))


if __name__ == "__main__":
    main()
