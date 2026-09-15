#!/usr/bin/env python3
"""Reduce the sealed full-B samples into one CPU-only acquisition diagnostic."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import mean


RAW_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-positive-branch-vs-repeat-event"
)
MANIFEST = RAW_ROOT / "input-preparation" / "candidate_manifest.json"
SIGNAL = RAW_ROOT / "signal-location-evidence.json"
OUTPUT = RAW_ROOT / "sampling-diagnostic-candidate.json"
SCRIPT = Path(__file__).resolve()

STRICT_THRESHOLD = 0.95
DIAGNOSTIC_THRESHOLDS = (0.25, 0.50, 0.75, 0.90, 0.95)
ROW_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
    r"<\|box_start\|>"
    r"<\|coord_(\d+)\|><\|coord_(\d+)\|>"
    r"<\|coord_(\d+)\|><\|coord_(\d+)\|>"
    r"<\|box_end\|>"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_description(value: str) -> str:
    return " ".join(value.casefold().split())


def iou_xyxy(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def parse_h_rows(text: str, width: int, height: int) -> list[dict]:
    rows = []
    for description, *raw_bins in ROW_RE.findall(text):
        bins = [int(value) for value in raw_bins]
        bbox = [
            round(bins[0] * width / 1000),
            round(bins[1] * height / 1000),
            round(bins[2] * width / 1000),
            round(bins[3] * height / 1000),
        ]
        rows.append(
            {
                "description": description,
                "description_normalized": normalize_description(description),
                "bbox_native_pixel_xyxy": bbox,
            }
        )
    if not rows:
        raise AssertionError("failed to parse literal h rows")
    return rows


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        name: round(at(fraction), 6)
        for name, fraction in (
            ("min", 0.0),
            ("q25", 0.25),
            ("median", 0.50),
            ("q75", 0.75),
            ("q90", 0.90),
            ("q95", 0.95),
            ("q99", 0.99),
            ("max", 1.0),
        )
    }


def overlap_summary(values: list[float]) -> dict:
    return {
        "quantiles": quantiles(values),
        "counts_above": {
            f">{threshold:.2f}": sum(value > threshold for value in values)
            for threshold in DIAGNOSTIC_THRESHOLDS
        },
        "fractions_above": {
            f">{threshold:.2f}": round(
                sum(value > threshold for value in values) / len(values), 6
            )
            for threshold in DIAGNOSTIC_THRESHOLDS
        },
    }


def top_counts(values: list[str], limit: int = 8) -> dict[str, int]:
    counter = Counter(values)
    items = counter.most_common(limit)
    result = dict(items)
    remainder = sum(counter.values()) - sum(result.values())
    if remainder:
        result["<other>"] = remainder
    return result


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["geometry_valid"]]
    if not valid:
        return {
            "sample_count": len(rows),
            "valid_count": 0,
            "outcome_counts": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        }
    any_h = [row["max_iou_any_h"] for row in valid]
    same_description_h = [row["max_iou_same_description_h"] for row in valid]
    greedy = [row["iou_greedy_repeat_row"] for row in valid]
    greedy_description = [row for row in valid if row["greedy_description_match"]]
    argmax_fractions = [row["argmax_target_fraction"] for row in rows]
    return {
        "sample_count": len(rows),
        "valid_count": len(valid),
        "outcome_counts": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "unique_valid_native_bboxes": len({tuple(row["bbox"]) for row in valid}),
        "description_counts": top_counts([row["description"] for row in valid]),
        "greedy_description_match": {
            "count": len(greedy_description),
            "fraction_of_valid": round(len(greedy_description) / len(valid), 6),
        },
        "description_matches_any_h": {
            "count": sum(row["description_matches_any_h"] for row in valid),
            "fraction_of_valid": round(
                sum(row["description_matches_any_h"] for row in valid) / len(valid), 6
            ),
        },
        "max_iou_to_any_prior_h": overlap_summary(any_h),
        "max_iou_to_same_description_prior_h": overlap_summary(same_description_h),
        "iou_to_exact_greedy_repeat_row": overlap_summary(greedy),
        "joint_greedy_description_and_spatial_nearness": {
            f">{threshold:.2f}": sum(
                row["greedy_description_match"]
                and row["iou_greedy_repeat_row"] > threshold
                for row in valid
            )
            for threshold in DIAGNOSTIC_THRESHOLDS
        },
        "action_score_projection": {
            "argmax_target_token_fraction_mean": round(mean(argmax_fractions), 6),
            "argmax_target_token_fraction_quantiles": quantiles(argmax_fractions),
            "action_mean_logprob_mean": round(mean(row["action_mean_logprob"] for row in rows), 6),
            "note": "Aggregate only; saved artifacts do not identify which token positions differed from argmax.",
        },
    }


def compact_summary(rows: list[dict]) -> dict:
    """Keep per-stratum evidence useful without reproducing the full reduction."""
    full = summarize(rows)
    if not full.get("valid_count"):
        return full

    def compact_overlap(name: str, thresholds: tuple[str, ...]) -> dict:
        overlap = full[name]
        return {
            "median": overlap["quantiles"]["median"],
            "q95": overlap["quantiles"]["q95"],
            "max": overlap["quantiles"]["max"],
            "counts_above": {
                threshold: overlap["counts_above"][threshold] for threshold in thresholds
            },
        }

    return {
        "sample_count": full["sample_count"],
        "valid_count": full["valid_count"],
        "outcome_counts": full["outcome_counts"],
        "unique_valid_native_bboxes": full["unique_valid_native_bboxes"],
        "description_counts": full["description_counts"],
        "greedy_description_match": full["greedy_description_match"],
        "max_iou_to_any_prior_h": compact_overlap(
            "max_iou_to_any_prior_h", (">0.25", ">0.50", ">0.90", ">0.95")
        ),
        "iou_to_exact_greedy_repeat_row": compact_overlap(
            "iou_to_exact_greedy_repeat_row", (">0.25", ">0.50", ">0.90", ">0.95")
        ),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    signal = json.loads(SIGNAL.read_text())
    signal_by_key = {row["candidate_id"]: row for row in signal["conditions"]}
    cases = {}
    for item in manifest["positives"]:
        key = item["candidate_id"]
        h_rows = parse_h_rows(
            item["h"]["text"], item["image"]["image_width"], item["image"]["image_height"]
        )
        signal_row = signal_by_key[key]
        assert len(h_rows) == signal_row["h_complete_valid_rows"]
        greedy = signal_row["first_free_complete_row"]["max_iou_prior_h_row"]
        cases[key] = {
            "case_id": item["case_id"],
            "h_rows": h_rows,
            "greedy_repeat_description": greedy["description"],
            "greedy_repeat_bbox": greedy["bbox_native_pixel_xyxy"],
            "greedy_repeat_prior_index": greedy["index_zero_based"],
        }

    update_paths = sorted((RAW_ROOT / "full-B").glob("update-*.json"))
    assert len(update_paths) == 32
    rows = []
    seeds = set()
    update_hash_lines = []
    for expected_step, path in enumerate(update_paths, 1):
        record = json.loads(path.read_text())
        assert record["step"] == expected_step and record["arm"] == "B"
        update_hash_lines.append(f"{path.name}\t{sha256(path)}")
        assert len(record["ranks"]) == 8
        for rank_record in record["ranks"]:
            rank = rank_record["rank"]
            events = [item for item in rank_record["records"] if item["kind"] == "event"]
            assert len(events) == 3
            for item in events:
                sample = item["sample"]
                event = sample["event"]
                seed = sample["seed"]
                assert seed not in seeds
                seeds.add(seed)
                key = sample["candidate_id"]
                case = cases[key]
                row = {
                    "step": expected_step,
                    "rank": rank,
                    "candidate_id": key,
                    "seed": seed,
                    "outcome": event["outcome"],
                    "geometry_valid": bool(event["geometry_valid"]),
                    "action_mean_logprob": item["action_mean_logprob"],
                    "argmax_target_fraction": (
                        item["chosen_token_score"]["argmax_target_tokens"]
                        / item["chosen_token_score"]["token_count"]
                    ),
                }
                if row["geometry_valid"]:
                    bbox = event["candidate_bbox_pixel_xyxy"]
                    description = normalize_description(event["candidate_description"])
                    overlaps = [iou_xyxy(bbox, prior["bbox_native_pixel_xyxy"]) for prior in case["h_rows"]]
                    assert abs(max(overlaps) - event["max_iou_to_h"]) < 1e-12
                    same_description_overlaps = [
                        overlap
                        for overlap, prior in zip(overlaps, case["h_rows"], strict=True)
                        if prior["description_normalized"] == description
                    ]
                    row.update(
                        {
                            "bbox": bbox,
                            "description": description,
                            "max_iou_any_h": max(overlaps),
                            "max_iou_same_description_h": max(same_description_overlaps, default=0.0),
                            "iou_greedy_repeat_row": iou_xyxy(bbox, case["greedy_repeat_bbox"]),
                            "greedy_description_match": description
                            == normalize_description(case["greedy_repeat_description"]),
                            "description_matches_any_h": any(
                                prior["description_normalized"] == description for prior in case["h_rows"]
                            ),
                        }
                    )
                rows.append(row)

    assert len(rows) == 768 and len(seeds) == 768
    assert Counter(row["outcome"] for row in rows) == Counter(
        {"valid_nonduplicate": 751, "geometry_invalid": 10, "malformed": 6, "eos": 1}
    )
    assert all(row.get("max_iou_any_h", 0.0) <= STRICT_THRESHOLD for row in rows)

    source_set_sha = hashlib.sha256(("\n".join(update_hash_lines) + "\n").encode()).hexdigest()
    output = {
        "schema": "repeat_recovery_sampling_diagnostic.v1",
        "status": "candidate",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "question": (
            "Do raw-softmax rows miss the exact greedy repeat because complete-action mass is diffuse, "
            "or do they mostly remain spatially near the same prior owner under coordinate jitter or description aliases?"
        ),
        "scope": {
            "model_calls": 0,
            "gpu_calls": 0,
            "sample_count": len(rows),
            "steps": [1, 32],
            "ranks_per_step": 8,
            "conditions": sorted(cases),
            "surface": "sealed full-B raw-softmax first retained actions under fixed literal h",
        },
        "metric_boundary": {
            "decision_event": {
                "metric": "class-blind native-pixel IoU to any earlier valid h row",
                "operator": ">",
                "threshold": STRICT_THRESHOLD,
                "changed": False,
            },
            "looser_thresholds": {
                "role": "descriptive spatial-nearness diagnostics only; never same-owner gold",
                "values": list(DIAGNOSTIC_THRESHOLDS[:-1]),
            },
        },
        "sources": {
            "candidate_manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST)},
            "signal_location_evidence": {"path": str(SIGNAL), "sha256": sha256(SIGNAL)},
            "full_B_update_files": {
                "directory": str(RAW_ROOT / "full-B"),
                "count": len(update_paths),
                "sha256_of_name_and_content_sha256_lines": source_set_sha,
            },
            "reducer": {"path": str(SCRIPT), "sha256": sha256(SCRIPT)},
        },
        "greedy_repeat_anchors": {
            key: {
                "case_id": case["case_id"],
                "description": case["greedy_repeat_description"],
                "bbox_native_pixel_xyxy": case["greedy_repeat_bbox"],
                "prior_h_row_index_zero_based": case["greedy_repeat_prior_index"],
            }
            for key, case in cases.items()
        },
        "aggregate": summarize(rows),
        "stable50_initial_step": summarize([row for row in rows if row["step"] == 1]),
        "stable50_initial_step_per_condition": {
            key: compact_summary(
                [
                    row
                    for row in rows
                    if row["step"] == 1 and row["candidate_id"] == key
                ]
            )
            for key in sorted(cases)
        },
        "later_changing_parameters_steps_2_to_32": compact_summary(
            [row for row in rows if row["step"] > 1]
        ),
        "per_condition": {
            key: compact_summary([row for row in rows if row["candidate_id"] == key])
            for key in sorted(cases)
        },
        "interpretation_boundary": {
            "observation": (
                "All strict D events remain zero. Exact-description agreement and class-blind overlap are "
                "reported separately so description aliases cannot hide a spatially near prior box."
            ),
            "supported_inference_rule": (
                "A small near-threshold tail weakens simple threshold-edge jitter; it cannot exclude severe "
                "reboxing of one physical owner, because no new owner labels or visual judgments are introduced."
            ),
            "not_established": [
                "physical-owner identity for sampled rows",
                "a KV, copying, cursor, or explicit owner-memory mechanism",
                "stationary sampling probability across steps 1 through 32",
                "original-prompt model quality",
            ],
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "aggregate": output["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
