"""Build and validate the bounded paired review for image 219546."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


BASE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-14-training-set-completion-curriculum"
)
OUT = BASE / "fourth-fit-owner-reviews-v1/image-000000219546"
INHERITANCE = BASE / "fourth-fit-owner-reviews-v1/match-inheritance-v2.json"
V4 = BASE / "target-owners-complete-v4.json"
V5 = BASE / "target-owners-complete-v5.json"

PHASES = {
    "parent64": {
        "step": 64,
        "packet": BASE / (
            "fourth-fit-eval-preparation-v1/parent-step-00064/"
            "image-000000219546/packet.json"
        ),
        "native": BASE / (
            "third-fit-v1/readback-recovery/rows/"
            "step-00064-image-000000219546.json"
        ),
    },
    "final256": {
        "step": 256,
        "packet": BASE / (
            "fourth-fit-eval-preparation-v1/fourth-step-00256/"
            "image-000000219546/packet.json"
        ),
        "native": BASE / (
            "fourth-fit-v1/readback-recovery/rows/"
            "step-00256-image-000000219546.json"
        ),
    },
}


UNMATCHED = {
    "parent64": {
        "p11": {
            "owner_id": "stable-new-219546-green-glass-bowl",
            "extent": "wrong",
            "class": "verified",
            "physical_status": "repeat",
            "reason": (
                "Viewed unmatched row: this is a later partial rebox of the green "
                "glass bowl already emitted at p6. It omits the left/top body, so "
                "it is a repeat with wrong extent and both CE targets masked."
            ),
        },
        "p25": {
            "owner_id": "stable-new-219546-oval-serving-bowl",
            "extent": "reasonable",
            "class": "wrong",
            "physical_status": "true_unique",
            "reason": (
                "Viewed unmatched row: the box coherently covers the full brown oval "
                "serving bowl and is the first occurrence of this owner; literal "
                "'fork' is wrong. Later inherited p27 is the route repeat."
            ),
        },
        "p28": {
            "owner_id": None,
            "extent": "wrong",
            "class": "unknown",
            "physical_status": "false",
            "reason": (
                "Viewed unmatched row: the singular bowl box encloses a stack of "
                "multiple distinct rectangular dishes rather than one atomic owner. "
                "It is not a full-box outside-v5 admission candidate."
            ),
        },
    },
    "final256": {
        "p37": {
            "owner_id": None,
            "extent": "wrong",
            "class": "unknown",
            "physical_status": "false",
            "reason": (
                "Viewed unmatched row: the broad box combines several physical "
                "owners (upper bowl, clear vessel, dark jar, grain bowl, and part of "
                "the transparent jar), so no single bowl owner or full geometry is "
                "present."
            ),
        },
        "p38": {
            "owner_id": "pending-new:219546:blue-vessel-visible-utensil",
            "extent": "reasonable",
            "class": "unknown",
            "physical_status": "true_unique",
            "reason": (
                "Viewed unmatched row: a distinct wooden serving utensil protrudes "
                "from the blue ceramic vessel and is absent from v5. The box covers "
                "its full visible extent, but the submerged head leaves spoon versus "
                "another utensil class unknown."
            ),
        },
    },
}


UNMATCHED_VIEW_NOTES = {
    "parent64": {
        "g0011": (
            "p11 binds the already-emitted green glass bowl, but only its right/lower "
            "portion; repeat, wrong extent, bowl literal visually supported."
        ),
        "g0025": (
            "p25 covers the complete brown oval serving bowl. It is the first route "
            "occurrence of that owner; the literal fork is visibly wrong."
        ),
        "g0028": (
            "p28 encloses several stacked rectangular dishes, not one atomic bowl; "
            "no outside-v5 full-box candidate is supported."
        ),
    },
    "final256": {
        "g0037": (
            "p37 is a compound multi-owner region. It does not match the full "
            "transparent-serving-jar reference [564,322,684,480] or any single bowl."
        ),
        "g0038": (
            "p38 is a separate visible wooden utensil emerging from the blue ceramic "
            "vessel. Its visible extent is complete; specific spoon class remains "
            "unknown because the head is occluded."
        ),
    },
}


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path, *, size: bool = False):
    value = {"path": str(path), "sha256": digest(path)}
    if size:
        value["size_bytes"] = path.stat().st_size
    return value


def make_sheets(phase: str, packet: dict) -> list[Path]:
    packet_dir = PHASES[phase]["packet"].parent
    rows = {r["prediction_id"]: r for r in packet["full_raw_rows"]}
    groups = packet["rendered"]["visual_groups"]
    outputs = []
    for start in range(0, len(groups), 4):
        chunk = groups[start : start + 4]
        end = start + len(chunk) - 1
        out = OUT / f"{phase}-view-sheet-{start:02d}-{end:02d}.jpg"
        sheet = Image.new("RGB", (1600, 1680), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        for offset, group in enumerate(chunk):
            y = offset * 420
            ids = group["member_prediction_ids"]
            row = rows[ids[0]]
            title = (
                f"{phase} {group['visual_group_id']} ids={','.join(ids)} "
                f"order={row['generated_order']} status={row['status']} "
                f"desc={row.get('description')} bins={row.get('coord_bins_1000')}"
            )
            draw.text((10, y + 5), title, fill="black")
            for column, key in enumerate(("context_crop", "tight_crop")):
                image = Image.open(group[key]["path"]).convert("RGB")
                image.thumbnail((780, 370), Image.Resampling.LANCZOS)
                x = 10 + column * 790 + (780 - image.width) // 2
                iy = y + 40 + (370 - image.height) // 2
                sheet.paste(image, (x, iy))
        sheet.save(out, quality=92)
        outputs.append(out)
    return outputs


def exact_reuse_source(packet: dict, pid: str):
    flat = next(d for d in packet["flat_decisions"] if d["prediction_id"] == pid)
    reuse = flat["exact_reuse"]
    if reuse["state"] != "exact_literal_reuse_candidate":
        return None
    source = reuse["source_candidates"][0]["source"]
    bindings = source["source_bindings"]
    return {
        "checkpoint_step": source["checkpoint_step"],
        "proposal_id": source["proposal_id"],
        "raw_source_row_sha256": source["raw_source_row_sha256"],
        "source_review_decision_sha256": source["source_review_decision_sha256"],
        "review_path": bindings["review"]["path"],
        "review_sha256": bindings["review"]["sha256"],
        "review_notes_path": bindings["review_notes"]["path"],
        "review_notes_sha256": bindings["review_notes"]["sha256"],
    }


def build_decisions(phase: str, packet: dict, rule: dict, v4_rows: list, v5_rows: list):
    v4 = {str(r["owner_id"]): r for r in v4_rows}
    v5 = {str(r["owner_id"]): r for r in v5_rows}
    inherited = {
        m["prediction_id"]: m
        for m in rule["fixed_v4_matches"] + rule["v5_additional_matches"]
    }
    groups = {
        pid: group
        for group in packet["rendered"]["visual_groups"]
        for pid in group["member_prediction_ids"]
    }
    packet_dir = PHASES[phase]["packet"].parent
    prelim = []
    for row in packet["full_raw_rows"]:
        pid = row["prediction_id"]
        group = groups[pid]
        evidence = [
            str(packet_dir / "original.jpg"),
            str(packet_dir / "raw-generated-overlay.png"),
            str(packet_dir / "target-catalog-overlay.png"),
            group["context_crop"]["path"],
            group["tight_crop"]["path"],
        ]
        if row["status"] != "parsed_valid":
            prelim.append(
                {
                    "prediction_id": pid,
                    "generated_order": row["generated_order"],
                    "visual_group_id": group["visual_group_id"],
                    "owner_id": None,
                    "physical_status": "invalid_output",
                    "extent": "unknown",
                    "class": "unknown",
                    "basis": "parser_invalid",
                    "reason": (
                        "Parser-invalid raw row preserved in generated order; no valid "
                        "physical geometry or class conclusion is available."
                    ),
                    "evidence_paths": evidence,
                }
            )
            continue
        if pid in inherited:
            match = inherited[pid]
            owner = match["reference_owner_id"]
            catalog_row = v5[owner]
            literal = row.get("description")
            category = catalog_row.get("category")
            reuse_source = exact_reuse_source(packet, pid)
            if category is not None:
                class_status = "verified" if literal == category else "wrong"
                class_basis = (
                    f"catalog category {category!r} and literal {literal!r} "
                    + ("agree" if class_status == "verified" else "disagree")
                )
            elif reuse_source is not None:
                flat = next(
                    d for d in packet["flat_decisions"] if d["prediction_id"] == pid
                )
                class_status = flat["exact_reuse"]["class"]
                class_basis = "packet-provided exact source-bound class proof applies"
            else:
                class_status = "unknown"
                class_basis = (
                    "catalog category is unset and no exact source-bound class proof applies"
                )
            value = {
                "prediction_id": pid,
                "generated_order": row["generated_order"],
                "visual_group_id": group["visual_group_id"],
                "owner_id": owner,
                "physical_status": "true_unique",
                "extent": "reasonable",
                "class": class_status,
                "basis": "iou_matched_inherited",
                "reason": (
                    "match-inheritance-v2 fixed-v4/v5 one-to-one IoU-0.5 match "
                    f"assigns owner {owner} at IoU {match['iou']:.6f} with extent "
                    f"reasonable; {class_basis}. Physical repetition is recomputed "
                    "from generated order."
                ),
                "evidence_paths": evidence,
            }
            if reuse_source is not None:
                value["reuse_source"] = reuse_source
            prelim.append(value)
            continue
        visual = UNMATCHED[phase][pid]
        prelim.append(
            {
                "prediction_id": pid,
                "generated_order": row["generated_order"],
                "visual_group_id": group["visual_group_id"],
                "owner_id": visual["owner_id"],
                "physical_status": visual["physical_status"],
                "extent": visual["extent"],
                "class": visual["class"],
                "basis": "visual_unmatched_review",
                "reason": visual["reason"],
                "evidence_paths": evidence
                + [str(packet_dir / "annotation-catalog-v5-overlay.png")],
            }
        )

    # Physical repetition is defined by owner identity and generated order, not
    # by which one-to-one row received the inherited catalog match.
    seen = set()
    for decision in prelim:
        owner = decision["owner_id"]
        if decision["physical_status"] in {"true_unique", "repeat"}:
            decision["physical_status"] = "repeat" if owner in seen else "true_unique"
            seen.add(owner)

    for decision in prelim:
        owner = decision["owner_id"]
        valid = decision["physical_status"] != "invalid_output"
        qualified = (
            valid
            and decision["physical_status"] in {"true_unique", "repeat"}
            and decision["extent"] == "reasonable"
        )
        decision["coverage_eligible"] = qualified and owner in v4
        decision["annotation_coverage_eligible"] = qualified and owner in v5
        bbox_positive = (
            valid
            and decision["physical_status"] == "true_unique"
            and decision["extent"] == "reasonable"
            and owner in v5
        )
        decision["direct_CE"] = {
            "bbox": "positive" if bbox_positive else "mask",
            "description": (
                "positive"
                if bbox_positive and decision["class"] == "verified"
                else "mask"
            ),
        }
    return prelim


def note_record(path: Path, interpretation: str, **extra):
    value = {
        "view": str(path),
        "sha256": digest(path),
        "interpretation": interpretation,
    }
    value.update(extra)
    return value


def make_notes(packets: dict, sheets: dict) -> Path:
    notes = []
    for phase in ("parent64", "final256"):
        packet_dir = PHASES[phase]["packet"].parent
        notes.extend(
            [
                note_record(
                    packet_dir / "original.jpg",
                    (
                        "Dense serving-table scene used only as the image-level physical "
                        "reference. Matched owner/extent assignments are governed by "
                        "match-inheritance-v2; unmatched rows are judged from this image."
                    ),
                    phase=phase,
                ),
                note_record(
                    packet_dir / "raw-generated-overlay.png",
                    (
                        "Raw overlay viewed with every prediction ID/order retained. "
                        "It establishes route ordering for repeat recomputation and locates "
                        "unmatched and parser-invalid groups."
                    ),
                    phase=phase,
                ),
                note_record(
                    packet_dir
                    / (
                        "target-catalog-overlay.png"
                        if phase == "parent64"
                        else "annotation-catalog-v5-overlay.png"
                    ),
                    (
                        "Catalog overlay viewed. Fixed-v4 and current-v5 contain the same "
                        "41 owners for image 219546; transparent serving jar full reference "
                        "is [564,322,684,480], and the blue ceramic vessel class remains unset."
                    ),
                    phase=phase,
                ),
            ]
        )
        for sheet in sheets[phase]:
            notes.append(
                note_record(
                    sheet,
                    (
                        "Context+tight crops were viewed before the v2 steering and are "
                        "preserved here. They are non-decision-bearing for inherited matched "
                        "owner/extent rows; unmatched groups receive separate notes below."
                    ),
                    phase=phase,
                )
            )
        packet = packets[phase]
        groups = {
            g["visual_group_id"]: g for g in packet["rendered"]["visual_groups"]
        }
        for gid, interpretation in UNMATCHED_VIEW_NOTES[phase].items():
            group = groups[gid]
            for key in ("context_crop", "tight_crop"):
                path = Path(group[key]["path"])
                notes.append(
                    note_record(
                        path,
                        interpretation,
                        phase=phase,
                        visual_group_id=gid,
                        crop_kind=key,
                    )
                )
    path = OUT / "review-notes.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in notes))
    return path


def build_review(
    phase: str,
    packet: dict,
    rule: dict,
    decisions: list,
    v4_rows: list,
    v5_rows: list,
    notes_path: Path,
    parent_covered: set[str] | None = None,
):
    v4 = {str(r["owner_id"]) for r in v4_rows}
    v5 = {str(r["owner_id"]) for r in v5_rows}
    covered = {
        d["owner_id"] for d in decisions if d["coverage_eligible"]
    }
    annotated = {
        d["owner_id"] for d in decisions if d["annotation_coverage_eligible"]
    }
    counts = Counter()
    for decision in decisions:
        counts[f"physical_{decision['physical_status']}"] += 1
        counts[f"class_{decision['class']}"] += 1
    native = read(PHASES[phase]["native"])
    summary = {
        "target_owner_count": len(v4),
        "covered_owner_ids": sorted(covered),
        "missing_owner_ids": sorted(v4 - covered),
        "covered_owner_count": len(covered),
        "missing_owner_count": len(v4 - covered),
        "annotation_target_owner_count": len(v5),
        "annotation_covered_owner_ids": sorted(annotated),
        "annotation_missing_owner_ids": sorted(v5 - annotated),
        "physical_repeat_row_count": counts["physical_repeat"],
        "confirmed_false_row_count": counts["physical_false"],
        "physical_unknown_row_count": counts["physical_unknown"],
        "invalid_output_row_count": counts["physical_invalid_output"],
        "class_wrong_row_count": counts["class_wrong"],
        "class_unknown_row_count": counts["class_unknown"],
        "raw_stop_reason": native["decode_stop_reason"],
        "cap_debt": native["decode_stop_reason"] != "im_end",
        "manual_reference_outcomes": {
            "stable-new-219546-transparent-serving-jar": (
                "missing; no row supports its full [564,322,684,480] reference; "
                "the broad final p37 compound is ineligible"
            ),
            "second-fit:new:219546:blue-ceramic-vessel": (
                "missing; no row covers its full reference and its catalog class is unknown"
            ),
        },
    }
    if parent_covered is not None:
        summary.update(
            {
                "retained_parent_owner_ids": sorted(parent_covered & covered),
                "lost_parent_owner_ids": sorted(parent_covered - covered),
                "newly_covered_fixed_owner_ids": sorted(covered - parent_covered),
            }
        )
    candidates = []
    if phase == "final256":
        packet_dir = PHASES[phase]["packet"].parent
        candidates.append(
            {
                "owner_id": "pending-new:219546:blue-vessel-visible-utensil",
                "status": "pending_root_admission",
                "reference_prediction_id": "p38",
                "source_visual_group_ids": ["g0038"],
                "physical_status": "true_unique",
                "extent": "reasonable",
                "class": "unknown",
                "category": None,
                "category_certainty": "unknown_specific_utensil_class",
                "physical_identity": (
                    "distinct wooden serving utensil protruding from the blue ceramic vessel"
                ),
                "full_reference_proposal_coord_bins_1000": [420, 207, 472, 268],
                "full_reference_proposal_bbox_pixel_xyxy": [484, 179, 544, 232],
                "coverage_effect": "none_until_root_admission",
                "reason": (
                    "Original, raw overlay, context crop, and tight crop confirm one "
                    "outside-v5 utensil with complete visible geometry. The submerged "
                    "head does not verify the generated spoon class."
                ),
                "evidence_paths": [
                    str(packet_dir / "original.jpg"),
                    str(packet_dir / "raw-generated-overlay.png"),
                    str(packet_dir / "annotation-catalog-v5-overlay.png"),
                    str(packet_dir / "crops/g0038-context.png"),
                    str(packet_dir / "crops/g0038-tight.png"),
                ],
            }
        )
    return {
        "schema": "fourth_fit_paired_owner_review.v1",
        "image_id": 219546,
        "checkpoint_step": PHASES[phase]["step"],
        "phase": phase,
        "source_packet": binding(PHASES[phase]["packet"]),
        "target_catalog": binding(V4, size=True),
        "annotation_catalog": binding(V5, size=True),
        "matching_inheritance": binding(INHERITANCE),
        "status": "candidate_ready",
        "raw_row_count": len(packet["full_raw_rows"]),
        "decisions": decisions,
        "summary": summary,
        "new_owner_candidates": candidates,
        "validation": {
            "all_raw_prediction_ids_preserved": True,
            "all_generated_orders_preserved": True,
            "visual_group_membership_exact": True,
            "fixed_v4_partition_exact": True,
            "current_v5_partition_exact": True,
            "coverage_eligibility_recomputed": True,
            "route_repeats_recomputed_from_owner_and_order": True,
            "direct_ce_masks_valid": True,
            "parser_invalid_rows_preserved": True,
            "source_hashes_verified": True,
            "evidence_paths_and_hashes_verified": True,
            "native_eos_and_cap_verified": True,
            "matched_extent_source": "match-inheritance-v2 IoU-0.5",
            "unmatched_views_reviewed": sorted(UNMATCHED[phase]),
            "notes": binding(notes_path),
        },
    }


def validate_review(review: dict, packet: dict, rule: dict, v4_rows: list, v5_rows: list):
    assert review["source_packet"] == binding(PHASES[review["phase"]]["packet"])
    assert review["matching_inheritance"] == binding(INHERITANCE)
    assert review["target_catalog"]["sha256"] == digest(V4)
    assert review["annotation_catalog"]["sha256"] == digest(V5)
    raw = packet["full_raw_rows"]
    decisions = review["decisions"]
    assert [(r["prediction_id"], r["generated_order"]) for r in raw] == [
        (d["prediction_id"], d["generated_order"]) for d in decisions
    ]
    inherited = {
        m["prediction_id"]: m
        for m in rule["fixed_v4_matches"] + rule["v5_additional_matches"]
    }
    assert set(inherited) | set(rule["unmatched_valid_prediction_ids"]) | set(
        rule["parser_invalid_prediction_ids"]
    ) == {r["prediction_id"] for r in raw}
    groups = {
        pid: g["visual_group_id"]
        for g in packet["rendered"]["visual_groups"]
        for pid in g["member_prediction_ids"]
    }
    assert set(groups) == {r["prediction_id"] for r in raw}
    v4 = {str(r["owner_id"]) for r in v4_rows}
    v5 = {str(r["owner_id"]) for r in v5_rows}
    assert v4 <= v5
    seen = set()
    covered, annotated = set(), set()
    counts = Counter()
    for row, decision in zip(raw, decisions):
        pid = row["prediction_id"]
        assert decision["visual_group_id"] == groups[pid]
        if pid in inherited:
            assert decision["owner_id"] == inherited[pid]["reference_owner_id"]
            assert decision["extent"] == "reasonable"
            assert decision["basis"] == "iou_matched_inherited"
        physical = decision["physical_status"]
        assert (row["status"] == "parsed_valid") == (physical != "invalid_output")
        owner = decision["owner_id"]
        if physical in {"true_unique", "repeat"}:
            assert owner is not None
            assert physical == ("repeat" if owner in seen else "true_unique")
            seen.add(owner)
        qualified = (
            row["status"] == "parsed_valid"
            and physical in {"true_unique", "repeat"}
            and decision["extent"] == "reasonable"
        )
        assert decision["coverage_eligible"] == (qualified and owner in v4)
        assert decision["annotation_coverage_eligible"] == (qualified and owner in v5)
        if decision["coverage_eligible"]:
            covered.add(owner)
        if decision["annotation_coverage_eligible"]:
            annotated.add(owner)
        bbox_positive = qualified and physical == "true_unique" and owner in v5
        assert decision["direct_CE"] == {
            "bbox": "positive" if bbox_positive else "mask",
            "description": (
                "positive"
                if bbox_positive and decision["class"] == "verified"
                else "mask"
            ),
        }
        for evidence in decision["evidence_paths"]:
            path = Path(evidence)
            assert path.is_file()
            digest(path)
        counts[f"physical_{physical}"] += 1
        counts[f"class_{decision['class']}"] += 1
    summary = review["summary"]
    assert set(summary["covered_owner_ids"]) == covered
    assert set(summary["missing_owner_ids"]) == v4 - covered
    assert set(summary["annotation_covered_owner_ids"]) == annotated
    assert set(summary["annotation_missing_owner_ids"]) == v5 - annotated
    assert summary["target_owner_count"] == len(v4)
    assert summary["annotation_target_owner_count"] == len(v5)
    assert summary["covered_owner_count"] == len(covered)
    assert summary["missing_owner_count"] == len(v4 - covered)
    assert summary["physical_repeat_row_count"] == counts["physical_repeat"]
    assert summary["confirmed_false_row_count"] == counts["physical_false"]
    assert summary["physical_unknown_row_count"] == counts["physical_unknown"]
    assert summary["invalid_output_row_count"] == counts["physical_invalid_output"]
    assert summary["class_wrong_row_count"] == counts["class_wrong"]
    assert summary["class_unknown_row_count"] == counts["class_unknown"]
    for candidate in review["new_owner_candidates"]:
        assert candidate["full_reference_proposal_coord_bins_1000"]
        assert "category_certainty" in candidate
        for evidence in candidate["evidence_paths"]:
            assert Path(evidence).is_file()
            digest(Path(evidence))
    return covered


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inheritance = read(INHERITANCE)
    assert inheritance["threshold"] == 0.5
    packets = {phase: read(spec["packet"]) for phase, spec in PHASES.items()}
    sheets = {phase: make_sheets(phase, packets[phase]) for phase in PHASES}
    notes_path = make_notes(packets, sheets)
    v4_rows = [r for r in read(V4)["records"] if r["image_id"] == 219546]
    v5_rows = [r for r in read(V5)["records"] if r["image_id"] == 219546]
    assert len(v4_rows) == len(v5_rows) == 41
    rules = {
        phase: next(
            r
            for r in inheritance["rows"]
            if r["image_id"] == 219546 and r["phase"] == phase
        )
        for phase in PHASES
    }
    decisions = {
        phase: build_decisions(phase, packets[phase], rules[phase], v4_rows, v5_rows)
        for phase in PHASES
    }
    parent = build_review(
        "parent64", packets["parent64"], rules["parent64"], decisions["parent64"],
        v4_rows, v5_rows, notes_path
    )
    parent_covered = validate_review(
        parent, packets["parent64"], rules["parent64"], v4_rows, v5_rows
    )
    parent_path = OUT / "parent64-review.json"
    parent_path.write_text(json.dumps(parent, indent=2, sort_keys=False) + "\n")
    final = build_review(
        "final256", packets["final256"], rules["final256"], decisions["final256"],
        v4_rows, v5_rows, notes_path, parent_covered=parent_covered
    )
    final_covered = validate_review(
        final, packets["final256"], rules["final256"], v4_rows, v5_rows
    )
    assert set(final["summary"]["retained_parent_owner_ids"]) == parent_covered & final_covered
    assert set(final["summary"]["lost_parent_owner_ids"]) == parent_covered - final_covered
    assert set(final["summary"]["newly_covered_fixed_owner_ids"]) == final_covered - parent_covered
    final_path = OUT / "final256-review.json"
    final_path.write_text(json.dumps(final, indent=2, sort_keys=False) + "\n")
    print(
        json.dumps(
            {
                "status": "validated",
                "image_id": 219546,
                "parent64": {
                    "raw_rows": len(parent["decisions"]),
                    "covered": parent["summary"]["covered_owner_count"],
                    "missing": parent["summary"]["missing_owner_count"],
                    "repeat": parent["summary"]["physical_repeat_row_count"],
                    "false": parent["summary"]["confirmed_false_row_count"],
                    "invalid": parent["summary"]["invalid_output_row_count"],
                },
                "final256": {
                    "raw_rows": len(final["decisions"]),
                    "covered": final["summary"]["covered_owner_count"],
                    "missing": final["summary"]["missing_owner_count"],
                    "repeat": final["summary"]["physical_repeat_row_count"],
                    "false": final["summary"]["confirmed_false_row_count"],
                    "invalid": final["summary"]["invalid_output_row_count"],
                    "retained": len(final["summary"]["retained_parent_owner_ids"]),
                    "lost": len(final["summary"]["lost_parent_owner_ids"]),
                    "new": len(final["summary"]["newly_covered_fixed_owner_ids"]),
                    "new_owner_candidates": len(final["new_owner_candidates"]),
                },
                "notes": binding(notes_path),
                "reviews": [binding(parent_path), binding(final_path)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
