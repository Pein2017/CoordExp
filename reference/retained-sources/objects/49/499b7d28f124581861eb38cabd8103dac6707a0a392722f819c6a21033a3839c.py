#!/usr/bin/env python3
"""Build and validate the parent step-16 qualified physical-owner ledger against target v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OUT = B / "parent16-v3-physical-ledger-v2"
DECISIONS = B / "first-fit-review-extraction-v1/decisions.jsonl"
EXTRACTION_ACCEPTANCE = B / "first-fit-review-extraction-v1/root-acceptance.json"
ACCEPTED_SUMMARY = B / "first-fit-review-extraction-v1/summary-v2.json"
TARGET_V2 = B / "target-owners-complete-v2.json"
TARGET_V3 = B / "target-owners-complete-v3.json"
ADMISSIONS = B / "first-fit-new-owner-admissions-v1/admissions.json"

EXPECTED = {
    DECISIONS: "8dcab2b7a25d82e2a871bc80d9a903485e599e4b44d2aab80bf7e765ba9dc85f",
    EXTRACTION_ACCEPTANCE: "9917a3e1da5743319d7964eaec0dafb0b22c5e6242cc27983aeafa786e7ffd2d",
    ACCEPTED_SUMMARY: "5fbb2e38f0dca8254af5bb8ff640f205964f7627d8a8699b830d18d640232048",
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


def row_projection(
    row: dict,
    line: int,
    effective_owner_id: str | None,
    basis: str,
    effective_extent: str,
    extent_authority: str,
    target_owner_ids: set[str],
) -> dict:
    checks = {
        "atomic_target_owner": effective_owner_id in target_owner_ids if effective_owner_id is not None else False,
        "physical_identity": row["physical_status"] in {"true_unique", "repeat"},
        "parser_valid": row["raw"]["status"] == "parsed_valid",
        "raw_axes_valid": row["raw"]["raw_axes_preserved"] is True,
        "extent_reasonable": effective_extent == "reasonable",
    }
    return {
        "proposal_id": row["proposal_id"],
        "prediction_id": row["prediction_id"],
        "source": {"path": str(DECISIONS), "line": line},
        "source_owner_id": row.get("owner_id"),
        "effective_owner_id": effective_owner_id,
        "owner_resolution_basis": basis,
        "physical_status": row["physical_status"],
        "canonical_extraction_extent": row["extent"],
        "effective_extent": effective_extent,
        "extent_authority": extent_authority,
        "class": row["class"],
        "raw_status": row["raw"]["status"],
        "raw_axes_preserved": row["raw"]["raw_axes_preserved"],
        "eligibility_checks": checks,
        "coverage_eligible_v3": all(checks.values()),
        "reason": row["reason"],
    }


def build() -> dict:
    primary = [binding(path, digest) for path, digest in EXPECTED.items()]
    extraction_acceptance = load_json(EXTRACTION_ACCEPTANCE)
    accepted_summary = load_json(ACCEPTED_SUMMARY)
    target_v2 = load_json(TARGET_V2)
    target_v3 = load_json(TARGET_V3)
    admissions = load_json(ADMISSIONS)
    numbered_rows = [(i, json.loads(line)) for i, line in enumerate(DECISIONS.read_text().splitlines(), 1) if line.strip()]
    step16 = [(i, row) for i, row in numbered_rows if row["step"] == 16]

    assert extraction_acceptance["status"] == "lead_accepted_review_derived_comparison"
    assert accepted_summary["aggregate"]["covered_owner_union_by_step"]["16"] == 151
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
    admission_by_pid = {a["reference_proposal_id"]: a for a in admissions["entries"]}
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
    identity_present_evidence = defaultdict(lambda: defaultdict(list))
    covered_evidence = defaultdict(lambda: defaultdict(list))
    for line, row in step16:
        if row["proposal_id"] in step16_admission_map:
            effective_owner = step16_admission_map[row["proposal_id"]]
            basis = "same_proposal_root_admission"
        else:
            effective_owner = str(row["owner_id"]) if row.get("owner_id") is not None else None
            basis = "canonical_step16_owner_id" if effective_owner is not None else "unassigned"
        admission = admission_by_pid.get(row["proposal_id"])
        if admission is not None:
            effective_extent = admission["original_proposal_extent"]
            extent_authority = "later_root_admission"
        else:
            effective_extent = row["extent"]
            extent_authority = "root_canonical_extraction"
        projected = row_projection(
            row,
            line,
            effective_owner,
            basis,
            effective_extent,
            extent_authority,
            target_by_image[row["image_id"]],
        )
        rows_by_image[row["image_id"]].append(projected)
        if (
            effective_owner is not None
            and effective_owner in target_by_image[row["image_id"]]
            and row["physical_status"] in {"true_unique", "repeat"}
        ):
            identity_present_evidence[row["image_id"]][effective_owner].append(row["proposal_id"])
        if projected["coverage_eligible_v3"]:
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

    accepted_fixed_by_image = {
        int(item["image_id"]): set(map(str, item["covered_fixed_owner_ids"]))
        for item in accepted_summary["summaries"]
        if item["step"] == 16
    }

    image_ledgers = []
    for image_id in sorted(target_by_image):
        covered = set(covered_evidence[image_id])
        identity_present = set(identity_present_evidence[image_id])
        missing = target_by_image[image_id] - covered
        rows = rows_by_image[image_id]
        assert covered | missing == target_by_image[image_id] and not (covered & missing)
        assert covered <= identity_present <= target_by_image[image_id]
        assert covered & old_by_image[image_id] == accepted_fixed_by_image[image_id]
        image_ledgers.append({
            "image_id": image_id,
            "target_owner_count": len(target_by_image[image_id]),
            "target_owner_ids": sorted(target_by_image[image_id]),
            "covered_owner_count": len(covered),
            "covered_owner_ids": sorted(covered),
            "covered_old_fixed220_count": len(covered & old_by_image[image_id]),
            "covered_admitted_v3_count": len(covered & admitted_by_image[image_id]),
            "covered_evidence": {key: sorted(value) for key, value in sorted(covered_evidence[image_id].items())},
            "identity_present_owner_count": len(identity_present),
            "identity_present_owner_ids": sorted(identity_present),
            "identity_present_but_ineligible_owner_ids": sorted(identity_present - covered),
            "identity_present_evidence": {key: sorted(value) for key, value in sorted(identity_present_evidence[image_id].items())},
            "missing_owner_count": len(missing),
            "missing_owner_ids": sorted(missing),
            "physical_repeats": [r for r in rows if r["physical_status"] == "repeat"],
            "confirmed_false": [r for r in rows if r["physical_status"] == "false"],
            "physical_unknown": [r for r in rows if r["physical_status"] == "unknown"],
            "class_debt_wrong": [r for r in rows if r["class"] == "wrong"],
            "class_unknown": [r for r in rows if r["class"] == "unknown"],
            "extent_debt_wrong": [r for r in rows if r["effective_extent"] == "wrong"],
            "extent_unknown": [r for r in rows if r["effective_extent"] == "unknown"],
            "all_step16_rows": rows,
        })

    status_counts = Counter(row["physical_status"] for _, row in step16)
    class_counts = Counter(row["class"] for _, row in step16)
    canonical_extent_counts = Counter(row["extent"] for _, row in step16)
    effective_extent_counts = Counter(row["effective_extent"] for rows in rows_by_image.values() for row in rows)
    summary = {
        "step16_row_count": len(step16),
        "image_count": len(image_ledgers),
        "target_v3_atomic_owner_count": len(v3_keys),
        "covered_old_fixed220_owner_count": sum(x["covered_old_fixed220_count"] for x in image_ledgers),
        "covered_newly_admitted_owner_count": sum(x["covered_admitted_v3_count"] for x in image_ledgers),
        "covered_physical_owner_count": sum(x["covered_owner_count"] for x in image_ledgers),
        "qualified_covered_owner_count": sum(x["covered_owner_count"] for x in image_ledgers),
        "identity_present_owner_count": sum(x["identity_present_owner_count"] for x in image_ledgers),
        "identity_present_but_ineligible_owner_count": sum(len(x["identity_present_but_ineligible_owner_ids"]) for x in image_ledgers),
        "missing_physical_owner_count": sum(x["missing_owner_count"] for x in image_ledgers),
        "qualified_missing_owner_count": sum(x["missing_owner_count"] for x in image_ledgers),
        "parser_invalid_row_count": sum(row["raw"]["status"] != "parsed_valid" for _, row in step16),
        "raw_axes_invalid_row_count": sum(row["raw"]["raw_axes_preserved"] is not True for _, row in step16),
        "physical_status_row_counts": dict(sorted(status_counts.items())),
        "class_row_counts": dict(sorted(class_counts.items())),
        "canonical_extraction_extent_row_counts": dict(sorted(canonical_extent_counts.items())),
        "effective_extent_row_counts": dict(sorted(effective_extent_counts.items())),
        "physical_repeat_row_count": status_counts["repeat"],
        "confirmed_false_row_count": status_counts["false"],
        "physical_unknown_row_count": status_counts["unknown"],
        "class_debt_wrong_row_count": class_counts["wrong"],
        "class_unknown_row_count": class_counts["unknown"],
        "extent_debt_wrong_row_count": effective_extent_counts["wrong"],
        "extent_unknown_row_count": effective_extent_counts["unknown"],
    }
    assert summary["covered_old_fixed220_owner_count"] == 151
    assert summary["covered_newly_admitted_owner_count"] == 4
    assert summary["covered_physical_owner_count"] == 155
    assert summary["qualified_covered_owner_count"] == 155
    assert summary["identity_present_owner_count"] == 161
    assert summary["identity_present_but_ineligible_owner_count"] == 6
    assert summary["missing_physical_owner_count"] == 73
    assert summary["qualified_missing_owner_count"] == 73
    assert summary["parser_invalid_row_count"] == 0
    assert summary["raw_axes_invalid_row_count"] == 0
    assert summary["covered_physical_owner_count"] + summary["missing_physical_owner_count"] == 228
    assert summary["physical_status_row_counts"] == {"false": 4, "repeat": 9, "true_unique": 167, "unknown": 21}
    assert summary["class_row_counts"] == {"unknown": 16, "verified": 174, "wrong": 11}
    assert summary["canonical_extraction_extent_row_counts"] == {"reasonable": 163, "unknown": 20, "wrong": 18}
    assert summary["effective_extent_row_counts"] == {"reasonable": 162, "unknown": 21, "wrong": 18}
    assert next(x for x in image_ledgers if x["image_id"] == 99937)["covered_owner_count"] == 5
    assert next(x for x in image_ledgers if x["image_id"] == 528944)["covered_owner_count"] == 2

    return {
        "schema": "training_set_completion.parent16_v3_physical_ledger.v2",
        "status": "candidate",
        "scope": "Actual first-fit step16 outputs for all 11 original training images, physically projected onto the 228-owner target v3.",
        "policy": {
            "physical_coverage": "A target owner is covered when a step16 row has its reviewed atomic identity, physical true_unique/repeat status, parsed-valid raw row, preserved raw axes, and effective reasonable extent.",
            "separation": "Class errors remain separate and do not remove otherwise qualified physical-owner coverage. Identity presence without reasonable extent is reported separately and is not coverage.",
            "cross_step": "No geometry, IoU, coordinate proximity, or description assigns an owner across step16 and step32.",
            "crowd": "Crowd owners are outside the 228 atomic denominator.",
            "excluded_source": "stage03-mask-preparation-v2 is a mixed teaching-prefix projection and is not used for parent16 counts.",
        },
        "summary": summary,
        "cross_step_admission_checks": cross_step_checks,
        "post_extraction_ruling_note": {
            "proposal_id": "first-fit:step16:image-000000219546:p14",
            "identity_effect": "The later admission establishes the physical owner mapping used here.",
            "canonical_extraction_fields_for_trace_only": {"extent": "reasonable", "class": "verified"},
            "later_admission_fields": {"original_proposal_extent": "unknown", "category": None, "class_policy": "mask_description"},
            "effective_ruling": {"extent": "unknown", "coverage_eligible_v3": False},
            "limitation": "The later root admission is authoritative for this ledger; the older reasonable extent appears only as a historical source field in the eligibility trace.",
        },
        "source_bindings": {
            "primary": primary,
            "embedded_review_packet_override_and_admission_sources": embedded_verified,
            "embedded_binding_count": len(embedded_verified),
        },
        "images": image_ledgers,
        "limitations": [
            "This is physical-owner plus reasonable-geometry accounting, not CE-eligibility, class-quality, or natural-completion accounting.",
            "The 73 missing owners are absent from qualified step16 coverage; unmatched rows are not converted to false predictions.",
            "Three step32-only admissions have no source-explicit step16 alias and therefore add no parent16 coverage.",
            "The canonical p14 row for image219546 and the later admission differ on extent; the later root admission controls effective eligibility.",
        ],
    }


def validate(ledger: dict) -> None:
    assert ledger["schema"] == "training_set_completion.parent16_v3_physical_ledger.v2"
    summary = ledger["summary"]
    assert summary["target_v3_atomic_owner_count"] == 228
    assert summary["covered_physical_owner_count"] == 155
    assert summary["missing_physical_owner_count"] == 73
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
        assert set(image["covered_owner_ids"]) <= set(image["identity_present_owner_ids"])
        assert all(row["coverage_eligible_v3"] == all(row["eligibility_checks"].values()) for row in image["all_step16_rows"])
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
        trace_path = OUT / "eligibility-trace.jsonl"
        trace_rows = [row for image in ledger["images"] for row in image["all_step16_rows"]]
        trace_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trace_rows))
        receipt = {
            "schema": "training_set_completion.parent16_v3_physical_ledger_receipt.v2",
            "status": "candidate",
            "ledger": binding(ledger_path),
            "builder": binding(Path(__file__)),
            "eligibility_trace": binding(trace_path),
            "summary": ledger["summary"],
            "cross_step_admission_checks": ledger["cross_step_admission_checks"],
            "counterexample_replay": {
                "fixed220_exact_accepted_owner_sets": True,
                "fixed220_qualified_covered": 151,
                "image99937_qualified": {"covered": 5, "target": 8},
                "image528944_qualified": {"covered": 2, "target": 10},
                "image219546_p14": {"effective_extent": "unknown", "coverage_eligible_v3": False},
            },
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
        trace_path = OUT / "eligibility-trace.jsonl"
        assert binding(trace_path, receipt["eligibility_trace"]["sha256"])["size_bytes"] == receipt["eligibility_trace"]["size_bytes"]
        trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        assert trace_rows == [row for image in ledger["images"] for row in image["all_step16_rows"]]
        assert receipt["summary"] == ledger["summary"]
        print(json.dumps({"status": "valid", "summary": ledger["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
