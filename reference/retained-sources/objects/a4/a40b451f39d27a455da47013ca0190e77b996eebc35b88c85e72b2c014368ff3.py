#!/usr/bin/env python3
"""CPU-only literal checks for the repeat-multiplicity prefix proposal.

This script reads the immutable small-owner-repeat-origin census and writes
nothing.  It validates row provenance, canonical row shape, equal lengths,
same A/B/C row set, reversal pairing, and the common final C row.  It does
not load a model, use GT, or launch an inference.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


R = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-small-owner-repeat-origin/census/census.json"
)
EXPECTED_R_SHA256 = (
    "664856efcf2555097a419011f6f9485c87ebe3978a1064e2d78d0151444a447a"
)

# Raw-row indices are frozen source provenance, not a new row selector.
CANDIDATES: dict[int, dict[str, int]] = {
    9813: {"A": 0, "B": 6, "C": 10},
    158044: {"A": 13, "B": 16, "C": 20},
    417044: {"A": 1, "B": 2, "C": 4},
    502725: {"A": 0, "B": 2, "C": 3},
}

EXPECTED_RAW_SHA256: dict[int, dict[str, str]] = {
    9813: {
        "A": "effbba6ca754364be012c09634595b76f79ab2a6525cc227493cb389a1bb64b1",
        "B": "cbc74615003f9f403730aa6e4a8119458ed7c240dced2c032572ca7c03621cb8",
        "C": "7e37aa592269eb62c1d0f54fee8ad66b7ac4becb420370c42efe12a6934ff747",
    },
    158044: {
        "A": "fc03a6d102b10f37093531f525d65f361186ac7a2c18148f9c74a38189d0ef58",
        "B": "35c1709015d10c8e30a5d8965ef0a9d9fd9967d2c8f7e14901ae9acf9895a9d4",
        "C": "b373bf1a1b785b3512b12b353fbffad0468e65cc9b9ab398f5b79a2caab86267",
    },
    417044: {
        "A": "133b74655fdd8c245877fb99b8d3278d48e4770fc0b668fed9df7fa5d9d36f3c",
        "B": "94889e248781d1eabdf7403b79d706f57796a870cf71fc5d1623c46367fbb828",
        "C": "ac79545b64c79961c4ccd6ad00befcd747c99ca551ff74743082d3c1a4dbc0c9",
    },
    502725: {
        "A": "571bfd46821e0a2e37b76c62348f0cd468a2c039501fc7db27e0562e39d761eb",
        "B": "5a81bd782c936966d2cf8461054515981610be15740e901e680d61572e39e842",
        "C": "ef3d308cab946d79e71a816cef818ce47385d64dfd8b6782b98cb9555012d3e1",
    },
}

# Each four-row varied block has a reversal mate.  C is a one-row common
# suffix, so every arm contains the same unique source row set {A,B,C}.
PATTERNS: tuple[tuple[str, str], ...] = (
    ("alt_fwd", "ABAB"),
    ("alt_rev", "BABA"),
    ("Aheavy_fwd", "AAAB"),
    ("Aheavy_rev", "BAAA"),
    ("Bheavy_fwd", "BBBA"),
    ("Bheavy_rev", "ABBB"),
    ("block_fwd", "AABB"),
    ("block_rev", "BBAA"),
)

REVERSAL_PAIRS = (
    ("alt_fwd", "alt_rev"),
    ("Aheavy_fwd", "Aheavy_rev"),
    ("Bheavy_fwd", "Bheavy_rev"),
    ("block_fwd", "block_rev"),
)

EOS = 151645
PAD = 151643


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, separators=(",", ":")).encode())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pixel_iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def main() -> None:
    require(R.is_file(), f"missing immutable R census: {R}")
    observed_r_sha = sha256_bytes(R.read_bytes())
    require(observed_r_sha == EXPECTED_R_SHA256, "HOLD: R census hash changed")
    census = json.loads(R.read_text())
    by_image = {int(case["image_id"]): case for case in census["cases"]}
    require(set(CANDIDATES) <= set(by_image), "HOLD: a fixed candidate case is absent")

    arm_report: dict[str, Any] = {}
    for image_id, labels in CANDIDATES.items():
        case = by_image[image_id]
        rows = {int(row["raw_row_index"]): row for row in case["raw_rows"]}
        selected = {}
        for label, row_index in labels.items():
            require(row_index in rows, f"HOLD {image_id}: {label} source row absent")
            row = rows[row_index]
            require(
                row["raw_text_sha256"] == EXPECTED_RAW_SHA256[image_id][label],
                f"HOLD {image_id}: {label} source row digest changed",
            )
            require(row["native_status"] == "accepted", f"HOLD {image_id}/{label}: not accepted")
            require(row["geometry_valid"] is True, f"HOLD {image_id}/{label}: invalid geometry")
            require(row["complete_canonical_row"] is True, f"HOLD {image_id}/{label}: incomplete row")
            require(not row["strict_repeat_reference_row_indices"],
                    f"HOLD {image_id}/{label}: source row is itself a strict repeat")
            token_ids = row["token_ids"]
            token_texts = row["token_texts"]
            require(len(token_ids) == row["token_end_exclusive"] - row["token_start"],
                    f"HOLD {image_id}/{label}: token span/IDs mismatch")
            require(len(token_ids) == len(token_texts), f"HOLD {image_id}/{label}: text/ID mismatch")
            require(token_ids and EOS not in token_ids and PAD not in token_ids,
                    f"HOLD {image_id}/{label}: EOS/PAD in row")
            require(token_texts[0] == "<|object_ref_start|>" and
                    token_texts[-1] == "<|box_end|>",
                    f"HOLD {image_id}/{label}: non-canonical row wrappers")
            require(token_texts.count("<|box_start|>") == 1 and
                    token_texts.count("<|box_end|>") == 1 and
                    sum(text.startswith("<|coord_") for text in token_texts) == 4,
                    f"HOLD {image_id}/{label}: expected four coordinate tokens")
            require(isinstance(row["description"], str) and row["description"],
                    f"HOLD {image_id}/{label}: missing description")
            selected[label] = row

        descriptions = {row["description"] for row in selected.values()}
        require(len(descriptions) == 1, f"HOLD {image_id}: A/B/C descriptions differ")
        lengths = {len(row["token_ids"]) for row in selected.values()}
        require(len(lengths) == 1, f"HOLD {image_id}: A/B/C row lengths differ")
        row_length = next(iter(lengths))
        row_digests = {sha256_json(row["token_ids"]) for row in selected.values()}
        require(len(row_digests) == 3, f"HOLD {image_id}: A/B/C token rows are not distinct")

        pairwise_iou = {}
        for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
            value = pixel_iou(selected[left]["pixel_box_xyxy"], selected[right]["pixel_box_xyxy"])
            pairwise_iou[f"{left}{right}"] = value
            require(value < 0.95, f"HOLD {image_id}: {left}/{right} strict duplicate geometry")

        arm_details = {}
        for arm_name, block in PATTERNS:
            labels_in_arm = list(block) + ["C"]
            require(set(labels_in_arm) == {"A", "B", "C"},
                    f"HOLD {image_id}/{arm_name}: unique row set changed")
            flat: list[int] = []
            for label in labels_in_arm:
                flat.extend(selected[label]["token_ids"])
            require(len(flat) == 5 * row_length,
                    f"HOLD {image_id}/{arm_name}: prefix token count changed")
            require(flat[-row_length:] == selected["C"]["token_ids"],
                    f"HOLD {image_id}/{arm_name}: final C row changed")
            require(EOS not in flat and PAD not in flat,
                    f"HOLD {image_id}/{arm_name}: EOS/PAD in prefix")
            arm_details[arm_name] = {
                "block": block,
                "labels": labels_in_arm,
                "row_count": len(labels_in_arm),
                "prefix_token_count": len(flat),
                "multiplicity": dict(Counter(block)),
                "prefix_token_ids_sha256": sha256_json(flat),
            }

        by_name = {name: block for name, block in PATTERNS}
        for forward, reverse in REVERSAL_PAIRS:
            require(by_name[reverse] == by_name[forward][::-1],
                    f"HOLD {image_id}: {forward}/{reverse} is not a reversal pair")

        arm_report[str(image_id)] = {
            "description": next(iter(descriptions)),
            "source_rows": {
                label: {
                    "raw_row_index": labels[label],
                    "token_span": [selected[label]["token_start"], selected[label]["token_end_exclusive"]],
                    "token_count": len(selected[label]["token_ids"]),
                    "coord_bins": selected[label]["coord_bins"],
                    "pixel_box_xyxy": selected[label]["pixel_box_xyxy"],
                    "raw_text_sha256": selected[label]["raw_text_sha256"],
                    "strict_repeat_reference_row_indices": selected[label]["strict_repeat_reference_row_indices"],
                }
                for label in ("A", "B", "C")
            },
            "pairwise_pixel_iou": pairwise_iou,
            "row_token_count": row_length,
            "prefix_token_count": 5 * row_length,
            "arms": arm_details,
        }

    print(json.dumps({
        "status": "PASS_CPU_LITERAL_INVARIANTS",
        "claim_boundary": "synthetic conditional diagnosis only; no GT/owner or natural-quality claim",
        "source_census": str(R),
        "source_census_sha256": observed_r_sha,
        "candidate_cases": sorted(CANDIDATES),
        "conditional_arms_per_case": len(PATTERNS),
        "reversal_pairs": [list(pair) for pair in REVERSAL_PAIRS],
        "cases": arm_report,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(str(exc))
        raise SystemExit(2)
