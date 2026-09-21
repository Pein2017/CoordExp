#!/usr/bin/env python3
"""Build and validate the parent step-16 physical-owner ledger against target v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OUT = B / "parent16-v3-physical-ledger-v1"
DECISIONS = B / "first-fit-review-extraction-v1/decisions.jsonl"
EXTRACTION_ACCEPTANCE = B / "first-fit-review-extraction-v1/root-acceptance.json"
TARGET_V2 = B / "target-owners-complete-v2.json"
TARGET_V3 = B / "target-owners-complete-v3.json"
ADMISSIONS = B / "first-fit-new-owner-admissions-v1/admissions.json"

EXPECTED = {
    DECISIONS: "8dcab2b7a25d82e2a871bc80d9a903485e599e4b44d2aab80bf7e765ba9dc85f",
    EXTRACTION_ACCEPTANCE: "9917a3e1da5743319d7964eaec0dafb0b22c5e6242cc27983aeafa786e7ffd2d",
    TARGET_V2: "cd13fe4d064c1632fbb80143ea31783c3d18ea4ea849cd906cfbc80954931c63",
    TARGET_V3: "b869c14f35f764dbe5d8a9b594887ea4ffac48ae7c14009f810d340de1e18332",
    ADMISSIONS: "461b4f28ad3593abf08b8f1c68824c7e7fc9f0f58827d8e7b8c11568b3f1c8b8",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def binding(path: Path, expected: str | None = None) -> dict:
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise AssertionError(f"hash mismatch: {path}: {actual} != {expected}")
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def load_json(path: Path):
    return json.loads(path.read_text())


def row_projection(row: dict, line: int, effective_owner_id: str | None, basis: str) -> dict:
    return {
        "proposal_id": row["proposal_id"],
        "prediction_id": row["prediction_id"],
        "source": {"path": str(DECISIONS), "line": line},
        "source_owner_id": row.get("owner_id"),
        "effective_owner_id": effective_owner_id,
        "owner_resolution_basis": basis,
        "physical_status": row["physical_status"],
        "extent": row["extent"],
        "class": row["class"],
        "reason": row["reason"],
    }


def build() -> dict:
    primary = [binding(path, digest) for path, digest in EXPECTED.items()]
    extraction_acceptance = load_json(EXTRACTION_ACCEPTANCE)
    target_v2 = load_json(TARGET_V2)
    target_v3 = load_json(TARGET_V3)
    admissions = load_json(ADMISSIONS)
    numbered_rows = [(i, json.loads(line)) for i, line in enumerate(DECISIONS.read_text().splitlines(), 1) if line.strip()]
    step16 = [(i, row) for i, row in numbered_rows if row["step"] == 16]

    assert extraction_acceptance["status"] == "lead_accepted_review_derived_comparison"
    assert extraction_acceptance["canonical_rows"] == len(numbered_rows) == 392
    assert target_v2["atomic_target_count"] == len(target_v2["records"]) == 220
    assert target_v3["atomic_target_count"] == len(target_v3["records"]) == 228
    assert admissions["new_owner_count"] == len(admissions["entries"]) == 8
    assert len(step16) == 201
    assert {row["image_id"] for _, row in step16} == {record["image_id"] for record in target_v3["records"]}

    v2_keys = {(int(r["image_id"]), str(r["owner_id"])) for r in target_v2["records"]}
    v3_keys = {(int(r["image_id"]), str(r["owner_id"])) for r in target_v3["records"]}
    admission_keys = {(int(r["image_id"]), str(r["owner_id"])) for r in admissions["entries"]}
    assert not (v2_keys & admission_keys)
    assert v3_keys == v2_keys | admission_keys

    # Only same-row, root-admitted step-16 references are remapped. There is no
    # geometric or description-based assignment across checkpoints.
    step16_admission_map = {
        a["reference_proposal_id"]: str(a["owner_id"])
        for a in admissions["entries"]
        if ":step16:" in a["reference_proposal_id"]
    }
    assert len(step16_admission_map) == 5
    row_by_pid = {row["proposal_id"]: row for _, row in numbered_rows}
    for pid, owner_id in step16_admission_map.items():
        row = row_by_pid[pid]
        assert row["step"] == 16 and row["physical_status"] == "true_unique"
        assert (row["image_id"], owner_id) in admission_keys

    embedded_bindings = {}
    for _, row in step16:
        for source_key in ("source_packet", "source_review", "root_override_applied"):
            source = row.get(source_key)
            if source and source.get("path") and source.get("sha256"):
                embedded_bindings[(source["path"], source["sha256"])] = source_key
    for admission in admissions["entries"]:
        for source in admission["sources"]:
            embedded_bindings[(source["path"], source["sha256"])] = "admission_source"
    embedded_verified = []
    for (path_text, digest), role in sorted(embedded_bindings.items()):
        item = binding(Path(path_text), digest)
        item["role"] = role
        embedded_verified.append(item)

    target_by_image = defaultdict(set)
    old_by_image = defaultdict(set)
    admitted_by_image = defaultdict(set)
    for image_id, owner_id in v3_keys:
        target_by_image[image_id].add(owner_id)
    for image_id, owner_id in v2_keys:
        old_by_image[image_id].add(owner_id)
    for image_id, owner_id in admission_keys:
        admitted_by_image[image_id].add(owner_id)

    rows_by_image = defaultdict(list)
    covered_evidence = defaultdict(lambda: defaultdict(list))
    for line, row in step16:
        if row["proposal_id"] in step16_admission_map:
            effective_owner = step16_admission_map[row["proposal_id"]]
            basis = "same_proposal_root_admission"
        else:
            effective_owner = str(row["owner_id"]) if row.get("owner_id") is not None else None
            basis = "canonical_step16_owner_id" if effective_owner is not None else "unassigned"
        projected = row_projection(row, line, effective_owner, basis)
        rows_by_image[row["image_id"]].append(projected)
        if (
            effective_owner is not None
            and effective_owner in target_by_image[row["image_id"]]
            and row["physical_status"] in {"true_unique", "repeat"}
        ):
            covered_evidence[row["image_id"]][effective_owner].append(row["proposal_id"])

    # All three admissions whose evidence row is step32 lack a source-explicit
    # step16 alias. They therefore receive no coverage, but are never called a
    # false prediction. The ledger exposes the check instead of assigning by geometry.
    cross_step_checks = []
    for admission in admissions["entries"]:
        pid = admission["reference_proposal_id"]
        if ":step32:" not in pid:
            continue
        oid = str(admission["owner_id"])
        image_id = int(admission["image_id"])
        named_step16 = [
            row["proposal_id"]
            for row in rows_by_image[image_id]
            if row["source_owner_id"] == oid or row["effective_owner_id"] == oid
        ]
        assert not named_step16
        cross_step_checks.append({
            "image_id": image_id,
            "owner_id": oid,
            "admission_reference_proposal_id": pid,
            "step16_source_explicit_aliases": [],
            "disposition": "missing_from_verified_step16_coverage",
            "false_prediction_disposition": "none",
            "reason": "No canonical/reviewer/admission source explicitly links a step16 row to this owner; no geometric assignment was made.",
        })
    assert len(cross_step_checks) == 3

    image_ledgers = []
    for image_id in sorted(target_by_image):
        covered = set(covered_evidence[image_id])
        missing = target_by_image[image_id] - covered
        rows = rows_by_image[image_id]
        assert covered | missing == target_by_image[image_id] and not (covered & missing)
        image_ledgers.append({
            "image_id": image_id,
            "target_owner_count": len(target_by_image[image_id]),
            "target_owner_ids": sorted(target_by_image[image_id]),
            "covered_owner_count": len(covered),
            "covered_owner_ids": sorted(covered),
            "covered_old_fixed220_count": len(covered & old_by_image[image_id]),
            "covered_admitted_v3_count": len(covered & admitted_by_image[image_id]),
            "covered_evidence": {key: sorted(value) for key, value in sorted(covered_evidence[image_id].items())},
            "missing_owner_count": len(missing),
            "missing_owner_ids": sorted(missing),
            "physical_repeats": [r for r in rows if r["physical_status"] == "repeat"],
            "confirmed_false": [r for r in rows if r["physical_status"] == "false"],
            "physical_unknown": [r for r in rows if r["physical_status"] == "unknown"],
            "class_debt_wrong": [r for r in rows if r["class"] == "wrong"],
            "class_unknown": [r for r in rows if r["class"] == "unknown"],
            "extent_debt_wrong": [r for r in rows if r["extent"] == "wrong"],
            "extent_unknown": [r for r in rows if r["extent"] == "unknown"],
            "all_step16_rows": rows,
        })

    status_counts = Counter(row["physical_status"] for _, row in step16)
    class_counts = Counter(row["class"] for _, row in step16)
    extent_counts = Counter(row["extent"] for _, row in step16)
    summary = {
        "step16_row_count": len(step16),
        "image_count": len(image_ledgers),
        "target_v3_atomic_owner_count": len(v3_keys),
        "covered_old_fixed220_owner_count": sum(x["covered_old_fixed220_count"] for x in image_ledgers),
        "covered_newly_admitted_owner_count": sum(x["covered_admitted_v3_count"] for x in image_ledgers),
        "covered_physical_owner_count": sum(x["covered_owner_count"] for x in image_ledgers),
        "missing_physical_owner_count": sum(x["missing_owner_count"] for x in image_ledgers),
        "physical_status_row_counts": dict(sorted(status_counts.items())),
        "class_row_counts": dict(sorted(class_counts.items())),
        "extent_row_counts": dict(sorted(extent_counts.items())),
        "physical_repeat_row_count": status_counts["repeat"],
        "confirmed_false_row_count": status_counts["false"],
        "physical_unknown_row_count": status_counts["unknown"],
        "class_debt_wrong_row_count": class_counts["wrong"],
        "class_unknown_row_count": class_counts["unknown"],
        "extent_debt_wrong_row_count": extent_counts["wrong"],
        "extent_unknown_row_count": extent_counts["unknown"],
    }
    assert summary["covered_old_fixed220_owner_count"] == 156
    assert summary["covered_newly_admitted_owner_count"] == 5
    assert summary["covered_physical_owner_count"] == 161
    assert summary["missing_physical_owner_count"] == 67
    assert summary["covered_physical_owner_count"] + summary["missing_physical_owner_count"] == 228
    assert summary["physical_status_row_counts"] == {"false": 4, "repeat": 9, "true_unique": 167, "unknown": 21}
    assert summary["class_row_counts"] == {"unknown": 16, "verified": 174, "wrong": 11}
    assert summary["extent_row_counts"] == {"reasonable": 163, "unknown": 20, "wrong": 18}

    return {
        "schema": "training_set_completion.parent16_v3_physical_ledger.v1",
        "status": "candidate",
        "scope": "Actual first-fit step16 outputs for all 11 original training images, physically projected onto the 228-owner target v3.",
        "policy": {
            "physical_coverage": "A target owner is covered when a root-canonical step16 row is true_unique or repeat and has that exact target owner identity, including a same-proposal v3 admission mapping.",
            "separation": "Extent, class, repeat, false, and unknown rulings remain row-level and do not rewrite physical identity coverage.",
            "cross_step": "No geometry, IoU, coordinate proximity, or description assigns an owner across step16 and step32.",
            "crowd": "Crowd owners are outside the 228 atomic denominator.",
            "excluded_source": "stage03-mask-preparation-v2 is a mixed teaching-prefix projection and is not used for parent16 counts.",
        },
        "summary": summary,
        "cross_step_admission_checks": cross_step_checks,
        "post_extraction_ruling_note": {
            "proposal_id": "first-fit:step16:image-000000219546:p14",
            "identity_effect": "The later admission establishes the physical owner mapping used here.",
            "row_fields_preserved": {"extent": "reasonable", "class": "verified"},
            "later_admission_fields": {"original_proposal_extent": "unknown", "category": None, "class_policy": "mask_description"},
            "limitation": "The later admission is stricter for teaching eligibility, but this physical ledger preserves the root-canonical extraction row fields and exposes the difference instead of rewriting it.",
        },
        "source_bindings": {
            "primary": primary,
            "embedded_review_packet_override_and_admission_sources": embedded_verified,
            "embedded_binding_count": len(embedded_verified),
        },
        "images": image_ledgers,
        "limitations": [
            "This is physical owner accounting, not complete-box, CE-eligibility, class-quality, or natural-completion accounting.",
            "The 67 missing owners are absent from verified step16 physical coverage; unmatched rows are not converted to false predictions.",
            "Three step32-only admissions have no source-explicit step16 alias and therefore add no parent16 coverage.",
            "The canonical p14 row for image219546 and the later admission differ on extent/class teaching treatment; both rulings are retained explicitly.",
        ],
    }


def validate(ledger: dict) -> None:
    assert ledger["schema"] == "training_set_completion.parent16_v3_physical_ledger.v1"
    summary = ledger["summary"]
    assert summary["target_v3_atomic_owner_count"] == 228
    assert summary["covered_physical_owner_count"] == 161
    assert summary["missing_physical_owner_count"] == 67
    assert len(ledger["images"]) == 11
    assert sum(len(x["all_step16_rows"]) for x in ledger["images"]) == 201
    for image in ledger["images"]:
        target = set(image["target_owner_ids"])
        covered = set(image["covered_owner_ids"])
        missing = set(image["missing_owner_ids"])
        assert target == covered | missing and not covered & missing
        assert image["target_owner_count"] == len(target)
        assert image["covered_owner_count"] == len(covered)
        assert image["missing_owner_count"] == len(missing)
    for source in ledger["source_bindings"]["primary"] + ledger["source_bindings"]["embedded_review_packet_override_and_admission_sources"]:
        assert binding(Path(source["path"]), source["sha256"])["size_bytes"] == source["size_bytes"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.write == args.validate:
        parser.error("choose exactly one of --write or --validate")
    if args.write:
        ledger = build()
        validate(ledger)
        ledger_path = OUT / "ledger.json"
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        receipt = {
            "schema": "training_set_completion.parent16_v3_physical_ledger_receipt.v1",
            "status": "candidate",
            "ledger": binding(ledger_path),
            "builder": binding(Path(__file__)),
            "summary": ledger["summary"],
            "cross_step_admission_checks": ledger["cross_step_admission_checks"],
            "validation_command": f"python {Path(__file__)} --validate",
        }
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt["summary"], sort_keys=True))
    else:
        ledger = load_json(OUT / "ledger.json")
        validate(ledger)
        receipt = load_json(OUT / "receipt.json")
        assert binding(OUT / "ledger.json", receipt["ledger"]["sha256"])["size_bytes"] == receipt["ledger"]["size_bytes"]
        assert binding(Path(__file__), receipt["builder"]["sha256"])["size_bytes"] == receipt["builder"]["size_bytes"]
        assert receipt["summary"] == ledger["summary"]
        print(json.dumps({"status": "valid", "summary": ledger["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
