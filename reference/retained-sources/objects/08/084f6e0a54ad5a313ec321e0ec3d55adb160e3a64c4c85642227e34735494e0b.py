#!/usr/bin/env python3
"""CPU-only, fail-closed final-readback review-packet builder.

It renders the actual third-fit step-64 parent immediately.  On a later
explicit replay it consumes only complete, saved fourth-fit rows at 128/192/256;
it never starts a readback, model, or training process.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


HERE = Path(__file__).resolve().parent
B = HERE.parents[1]
WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
COHORT = (25274, 59571, 99937, 210457, 219546, 323322, 351017, 388795, 417044, 477415, 528944)
PARENT_STEP = 64
FOURTH_STEPS = (128, 192, 256)
ACQUISITION = B / "stage01-acquisition-v1-retry1-config-batch2/manifest.json"
TARGET = B / "target-owners-complete-v4.json"
PARENT_LEDGER = B / "parent16-v4-physical-ledger-v1/ledger.json"
ANNOTATIONS = B / "annotations-with-unlabeled-v3/manifest.json"
THIRD_RESULTS = B / "third-fit-review-extraction-v1/full-review-results.jsonl"
THIRD_SELECTOR_64 = B / "third-fit-selectors-v1/scored-step-64.json"
THIRD_ROWS = B / "third-fit-v1/readback-recovery/rows"
FOURTH_ROWS = B / "fourth-fit-v1/readback-recovery/rows"
RENDERER = B / "B/first-fit-visualization-preparation-v1/render_readback.py"


def canon(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canon(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canon(row) for row in rows))


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def renderer() -> Any:
    spec = importlib.util.spec_from_file_location("fourth_fit_packet_renderer", RENDERER)
    require(spec is not None and spec.loader is not None, f"cannot load renderer: {RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_content_hash(record: Mapping[str, Any]) -> str:
    return str(record["case"]["image_plan"]["image_content_sha256"])


def parser_state(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row.get("status"), row.get("drop_reason"), row.get("raw_span_sha256"))


def literal_key(content_hash: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Exact, order-independent key.  No normalization or synonym matching."""
    bins = row.get("coord_bins_1000")
    return (content_hash, tuple(bins) if isinstance(bins, list) else None, row.get("description"), *parser_state(row))


def crop_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Only identical raw renderings share a crop; every member remains listed."""
    return (*parser_state(row), row.get("description"), tuple(row.get("coord_bins_1000") or ()),
            tuple(row.get("bbox_pixel_xyxy") or ()))


def complete_rows(root: Path, step: int) -> list[Path] | None:
    found = sorted(root.glob(f"step-{step:05d}-image-*.json"))
    expected = {f"step-{step:05d}-image-{image_id:012d}.json" for image_id in COHORT}
    names = {path.name for path in found}
    if names != expected:
        return None
    rows = []
    for path in found:
        row = json.loads(path.read_text())
        require(int(row["image_id"]) in COHORT, f"out-of-cohort row: {path}")
        require(int(row.get("checkpoint_step", step)) == step, f"wrong checkpoint in {path}")
        rows.append(path)
    require(len({json.loads(path.read_text())["image_id"] for path in rows}) == len(COHORT), f"duplicate image rows at {step}")
    return rows


def selector_for_fourth(step: int, row_paths: list[Path], acquisition: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    """Use the accepted CPU selector implementation against already saved rows."""
    if str(WORKTREE) not in sys.path:
        sys.path.insert(0, str(WORKTREE))
    from probes.training_set_completion import readback_selectors as selectors

    readback = {"schema": "training_set_completion.fourth_fit_saved_readbacks.v1", "status": "saved_rows_complete",
                "checkpoint_step": step, "rows": [json.loads(path.read_text()) for path in row_paths]}
    tokenizer_root = Path(acquisition["model"]["base_model"]["root"])
    result = selectors.score_readback(readback, acquisition, target, selectors._load_tokenizer(tokenizer_root), cap=3084)
    result["checkpoint_step"] = step
    result["saved_row_sources"] = [binding(path) for path in row_paths]
    result["selector_boundary"] = "CPU parsing and geometry diagnostics only; no owner/class/physical conclusion."
    return result


def source_label_index(full_results: Iterable[Mapping[str, Any]], records: Mapping[int, Mapping[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    index: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in full_results:
        image_id = int(result["image_id"])
        raw = result["raw"]
        row = {"coord_bins_1000": raw.get("coord_bins_1000"), "description": raw.get("description"),
               "status": result.get("raw_status"), "drop_reason": raw.get("drop_reason"),
               "raw_span_sha256": result.get("raw_span_sha256")}
        semantic = {key: result.get(key) for key in ("owner_id", "extent", "class")}
        index[literal_key(image_content_hash(records[image_id]), row)].append({
            "semantic": semantic,
            "source": {"proposal_id": result["proposal_id"], "checkpoint_step": result["checkpoint_step"],
                       "raw_source_row_sha256": result["raw_source_row_sha256"],
                       "source_review_decision_sha256": result["source_review_decision_sha256"],
                       "source_bindings": result["source_bindings"]},
        })
    return index


def exact_reuse(row: Mapping[str, Any], index: Mapping[tuple[Any, ...], list[dict[str, Any]]], content_hash: str,
                owners_seen: set[str]) -> dict[str, Any]:
    candidates = index.get(literal_key(content_hash, row), [])
    semantics = {(x["semantic"]["owner_id"], x["semantic"]["extent"], x["semantic"]["class"]) for x in candidates}
    if len(semantics) > 1:
        return {"state": "fail_closed_ambiguous_conflicting_labels", "source_candidates": candidates,
                "physical_status_recomputed_from_current_order": "unknown"}
    if not candidates:
        return {"state": "no_exact_accepted_step32_geometry_label", "source_candidates": [],
                "physical_status_recomputed_from_current_order": "unknown"}
    semantic = candidates[0]["semantic"]
    owner = semantic["owner_id"]
    repeat = owner in owners_seen
    owners_seen.add(owner)
    return {"state": "exact_literal_reuse_candidate", "physical_owner_id": owner, "extent": semantic["extent"],
            "class": semantic["class"], "physical_status_recomputed_from_current_order": "repeat" if repeat else "true_unique",
            "source_candidates": candidates}


def render_packet(*, step: int, scored_row: Mapping[str, Any], records: Mapping[int, Mapping[str, Any]], target_by_image: Mapping[int, list[Mapping[str, Any]]],
                  label_index: Mapping[tuple[Any, ...], list[dict[str, Any]]], output: Path, source_selector: Path, rr: Any) -> dict[str, Any]:
    image_id = int(scored_row["image_id"])
    record = records[image_id]
    image_path = Path(record["case"]["image_path"]).resolve(strict=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    require([image.width, image.height] == [int(record["case"]["image_width"]), int(record["case"]["image_height"])], f"dimensions differ: {image_id}")
    refs = []
    for ref in target_by_image[image_id]:
        item = dict(ref)
        item["bbox_pixel_xyxy"] = rr.bins_to_pixels(item["reference_coord_bins_1000"], image.width, image.height)
        refs.append(item)
    raw = []
    for source_row in scored_row["raw_rows"]:
        item = dict(source_row)
        item["_raw_box_pixel_xyxy"] = rr.raw_box(item, image.width, image.height)
        raw.append(item)
    raw.sort(key=lambda x: (int(x.get("generated_order", 10**9)), str(x.get("prediction_id", ""))))
    packet_dir = output / f"image-{image_id:012d}"
    crops = packet_dir / "crops"
    crops.mkdir(parents=True, exist_ok=False)
    original = packet_dir / "original.jpg"
    shutil.copyfile(image_path, original)
    target_overlay = packet_dir / "target-catalog-overlay.png"
    raw_overlay = packet_dir / "raw-generated-overlay.png"
    rr.render_target_overlay(image, refs, target_overlay, image_id)
    rr.render_raw_overlay(image, raw, raw_overlay, image_id)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        groups[crop_signature(row)].append(row)
    group_for_prediction: dict[str, dict[str, Any]] = {}
    visual_groups = []
    for ordinal, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: (int(kv[1][0].get("generated_order", 10**9)), str(kv[1][0].get("prediction_id", ""))))):
        group_id = f"g{ordinal:04d}"
        representative = members[0]
        tight = crops / f"{group_id}-tight.png"
        context = crops / f"{group_id}-context.png"
        tight_frame = rr.render_crop(image, representative, tight, 8, "exact-signature-tight")
        context_frame = rr.render_crop(image, representative, context, 64, "exact-signature-context")
        group = {"visual_group_id": group_id, "member_prediction_ids": [str(x["prediction_id"]) for x in members],
                 "member_count": len(members), "exact_raw_signature": {"parser_state": list(parser_state(representative)),
                 "description": representative.get("description"), "coord_bins_1000": representative.get("coord_bins_1000"),
                 "bbox_pixel_xyxy": representative.get("bbox_pixel_xyxy")},
                 "tight_crop": {"path": str(tight.resolve()), "frame_pixel_xyxy": list(tight_frame), "sha256": sha(tight)},
                 "context_crop": {"path": str(context.resolve()), "frame_pixel_xyxy": list(context_frame), "sha256": sha(context)}}
        visual_groups.append(group)
        for member in members:
            group_for_prediction[str(member["prediction_id"])] = group
    owners_seen: set[str] = set()
    rendered = []
    decisions = []
    notes = []
    for row in raw:
        group = group_for_prediction[str(row["prediction_id"])]
        reuse = exact_reuse(row, label_index, image_content_hash(record), owners_seen)
        flat = {"schema": "training_set_completion.fourth_fit_eval_flat_decision.v1", "image_id": image_id, "checkpoint_step": step,
                "prediction_id": row["prediction_id"], "generated_order": row.get("generated_order"), "raw_parser_state": list(parser_state(row)),
                "literal_description": row.get("description"), "coord_bins_1000": row.get("coord_bins_1000"),
                "raw_span_sha256": row.get("raw_span_sha256"), "exact_reuse": reuse,
                "review_state": "pending_human_review; reuse is a source-bound candidate, never a new physical conclusion"}
        decisions.append(flat)
        notes.append({"schema": "training_set_completion.fourth_fit_eval_reviewer_note.v1", "image_id": image_id, "checkpoint_step": step,
                      "prediction_id": row["prediction_id"], "visual_group_id": group["visual_group_id"],
                      "note": "Open original, target overlay, raw overlay, then the shared exact-signature crops. Confirm or replace this candidate explicitly; do not infer a metric from this packet.",
                      "exact_reuse_state": reuse["state"], "source_candidate_count": len(reuse["source_candidates"])})
        rendered.append({"prediction_id": row["prediction_id"], "generated_order": row.get("generated_order"), "status": row.get("status"),
                         "drop_reason": row.get("drop_reason"), "description": row.get("description"), "coord_bins_1000": row.get("coord_bins_1000"),
                         "raw_bbox_pixel_xyxy": row.get("bbox_pixel_xyxy"), "render_box_pixel_xyxy": row.get("_raw_box_pixel_xyxy"),
                         "raw_span_sha256": row.get("raw_span_sha256"), "visual_group_id": group["visual_group_id"],
                         "shared_crop_paths": {"tight": group["tight_crop"]["path"], "context": group["context_crop"]["path"]}, "exact_reuse_state": reuse["state"]})
    packet = {"schema": "training_set_completion.fourth_fit_eval_packet.v1", "status": "review_ready_no_physical_conclusion", "image_id": image_id,
              "checkpoint_step": step, "route_id": scored_row.get("route_id"), "scored_row_sha256": hashlib.sha256(canon(scored_row)).hexdigest(),
              "source": {"scored_selector": binding(source_selector), "acquisition_manifest": binding(ACQUISITION), "target_catalog_v4": binding(TARGET),
                         "parent16_v4_physical_ledger": binding(PARENT_LEDGER), "third_fit32_full_review_results": binding(THIRD_RESULTS),
                         "original_image": binding(image_path), "image_content_sha256": image_content_hash(record), "dimensions": [image.width, image.height]},
              "target_catalog_references": [{"owner_id": x["owner_id"], "role": x.get("role"), "category": x.get("category"),
                                              "reference_coord_bins_1000": x["reference_coord_bins_1000"], "bbox_pixel_xyxy": x["bbox_pixel_xyxy"]} for x in refs],
              "full_raw_rows": raw,
              "rendered": {"original_image": {"path": str(original.resolve()), "sha256": sha(original)},
                           "target_catalog_overlay": {"path": str(target_overlay.resolve()), "sha256": sha(target_overlay)},
                           "raw_generated_overlay": {"path": str(raw_overlay.resolve()), "sha256": sha(raw_overlay)}, "visual_groups": visual_groups, "raw_rows": rendered},
              "flat_decisions": decisions, "reviewer_notes": notes,
              "counts": {"target_reference_count": len(refs), "raw_row_count": len(raw), "valid_prediction_count": sum(x.get("status") == "parsed_valid" for x in raw),
                         "raw_invalid_or_dropped_count": sum(x.get("status") != "parsed_valid" for x in raw), "visual_group_count": len(visual_groups),
                         "exact_reuse_candidate_count": sum(x["exact_reuse"]["state"] == "exact_literal_reuse_candidate" for x in decisions),
                         "unresolved_or_fail_closed_count": sum(x["exact_reuse"]["state"] != "exact_literal_reuse_candidate" for x in decisions)},
              "acceptance_boundary": {"all_raw_rows_preserved": True, "identical_raw_signatures_share_crops": True,
                                      "all_raw_member_ids_retained": True, "current_route_repetition_recomputed_from_order": True,
                                      "no_current_physical_conclusion": True, "no_metric_claim": True}}
    write(packet_dir / "packet.json", packet)
    write_jsonl(packet_dir / "flat-decisions.jsonl", decisions)
    write_jsonl(packet_dir / "reviewer-notes.jsonl", notes)
    return {"image_id": image_id, "packet": str((packet_dir / "packet.json").resolve()), "flat_decisions": str((packet_dir / "flat-decisions.jsonl").resolve()),
            "reviewer_notes": str((packet_dir / "reviewer-notes.jsonl").resolve()), "original": str(original.resolve()),
            "target_overlay": str(target_overlay.resolve()), "raw_overlay": str(raw_overlay.resolve()), **packet["counts"]}


def build_step(step: int, scored: Mapping[str, Any], *, records: Mapping[int, Mapping[str, Any]], target_by_image: Mapping[int, list[Mapping[str, Any]]],
               label_index: Mapping[tuple[Any, ...], list[dict[str, Any]]], rr: Any, output_root: Path, selector_path: Path) -> dict[str, Any]:
    require(not output_root.exists(), f"refusing to overwrite packet root: {output_root}")
    rows = {int(row["image_id"]): row for row in scored["rows"]}
    require(set(rows) == set(COHORT), f"selector cohort incomplete at {step}")
    packets = [render_packet(step=step, scored_row=rows[image_id], records=records, target_by_image=target_by_image,
                             label_index=label_index, output=output_root, source_selector=selector_path, rr=rr) for image_id in COHORT]
    return {"checkpoint_step": step, "packet_root": str(output_root.resolve()), "packets": packets,
            "raw_row_count": sum(x["raw_row_count"] for x in packets), "exact_reuse_candidate_count": sum(x["exact_reuse_candidate_count"] for x in packets),
            "unresolved_or_fail_closed_count": sum(x["unresolved_or_fail_closed_count"] for x in packets)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-future", action="store_true", help="Consume complete saved 128/192/256 rows if present; never wait for them.")
    args = parser.parse_args()
    require(HERE.exists(), "output root absent")
    acquisition, target, parent_ledger = (json.loads(path.read_text()) for path in (ACQUISITION, TARGET, PARENT_LEDGER))
    require(target["status"] == "lead-accepted" and target["atomic_target_count"] == 232, "fixed v4 target denominator changed")
    require(parent_ledger["status"] == "lead-accepted" and parent_ledger["summary"]["target_owner_count"] == 232, "parent v4 ledger changed")
    annotations = json.loads(ANNOTATIONS.read_text())
    require(annotations["counts"]["valid_unlabeled"] == 77, "annotation-v3 non-GT export changed")
    records = {int(row["image_id"]): row for row in acquisition["records"]}
    require(set(records) == set(COHORT), "acquisition cohort changed")
    target_by_image: dict[int, list[Mapping[str, Any]]] = {image_id: [] for image_id in COHORT}
    for ref in target["records"]:
        target_by_image[int(ref["image_id"])].append(ref)
    full_results = [json.loads(line) for line in THIRD_RESULTS.read_text().splitlines() if line]
    label_index = source_label_index(full_results, records)
    rr = renderer()
    parent_rows = complete_rows(THIRD_ROWS, PARENT_STEP)
    require(parent_rows is not None, "immediate parent step-64 rows incomplete")
    parent_scored = json.loads(THIRD_SELECTOR_64.read_text())
    selector_dir = HERE / "selectors"
    # v2 is canonical: it adds literal full raw rows and genuine JSONL review
    # logs after the first local rendering omitted those two preservation aids.
    parent_root = HERE / "parent-step-00064-v2"
    results: list[dict[str, Any]] = []
    if not parent_root.exists():
        results.append(build_step(PARENT_STEP, parent_scored, records=records, target_by_image=target_by_image, label_index=label_index, rr=rr,
                                  output_root=parent_root, selector_path=THIRD_SELECTOR_64))
    else:
        results.append({"checkpoint_step": PARENT_STEP, "state": "existing_packet_root_reused", "packet_root": str(parent_root.resolve())})
    pending = []
    for step in FOURTH_STEPS:
        row_paths = complete_rows(FOURTH_ROWS, step)
        packet_root = HERE / f"fourth-step-{step:05d}"
        if row_paths is None:
            pending.append({"checkpoint_step": step, "state": "pending_saved_rows_incomplete", "found_row_count": len(list(FOURTH_ROWS.glob(f"step-{step:05d}-image-*.json"))), "required_row_count": len(COHORT)})
            continue
        selector_path = selector_dir / f"scored-step-{step:05d}.json"
        if not selector_path.exists():
            require(args.build_future, f"complete fourth-fit rows found at {step}; rerun with --build-future to parse CPU-only")
            selector = selector_for_fourth(step, row_paths, acquisition, target)
            write(selector_dir / f"readback-step-{step:05d}.json", {"checkpoint_step": step, "rows": [binding(path) for path in row_paths]})
            write(selector_path, selector)
        if not packet_root.exists():
            scored = json.loads(selector_path.read_text())
            results.append(build_step(step, scored, records=records, target_by_image=target_by_image, label_index=label_index, rr=rr,
                                      output_root=packet_root, selector_path=selector_path))
        else:
            results.append({"checkpoint_step": step, "state": "existing_packet_root_reused", "packet_root": str(packet_root.resolve())})
    mapping = {"schema": "training_set_completion.fourth_fit_exact_reuse_mapping.v1", "status": "candidate_source_mapping_ready",
               "key": ["image_content_sha256", "coord_bins_1000", "literal_description", "raw_parser_state"],
               "source_bindings": {"third_fit32_full_review_results": binding(THIRD_RESULTS), "parent16_v4_physical_ledger": binding(PARENT_LEDGER)},
               "entry_count": sum(len(x) for x in label_index.values()), "unique_literal_key_count": len(label_index),
               "ambiguously_labeled_key_count": sum(len({(x['semantic']['owner_id'], x['semantic']['extent'], x['semantic']['class']) for x in v}) > 1 for v in label_index.values()),
               "policy": "Reuse only exact source-bound owner/extent/class candidates. Any conflicting label fails closed. Current route repetition is computed from current generated order."}
    write(HERE / "exact-reuse-mapping.json", mapping)
    pending_receipt = {"schema": "training_set_completion.fourth_fit_final_eval_pending.v1", "status": "pending_final_saved_rows" if pending else "all_saved_rows_consumed",
                       "fixed_metric_target": {"catalog_version": "v4", "atomic_target_count": 232, "source": binding(TARGET)},
                       "annotation_only_export": {"source": binding(ANNOTATIONS), "valid_non_gt_annotation_count": 77, "new_annotation_count_user_fixed": 14,
                                                  "metric_denominator_policy": "annotation export is review context only and never changes the frozen v4 denominator"},
                       "parent_actual_step": 64, "canonical_parent_packet_root": str(parent_root.resolve()),
                       "superseded_local_packet_root": str((HERE / "parent-step-00064").resolve()) if (HERE / "parent-step-00064").exists() else None,
                       "future_steps": list(FOURTH_STEPS), "pending": pending, "built": results,
                       "replay": f"python {HERE / 'build_final_eval_packets.py'} --build-future", "no_training_or_gpu_work": True,
                       "no_physical_conclusions": True}
    write(HERE / "pending-or-receipt.json", pending_receipt)
    print(json.dumps({"pending": pending, "built": results, "replay": pending_receipt["replay"]}, indent=2))


if __name__ == "__main__":
    main()
