#!/usr/bin/env python3
"""Build narrowly scoped step-64 introduced-owner visual-review packets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

WORKTREE = Path("/data/CoordExp/.worktrees/research-probes")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from probes.training_set_completion.readback_selectors import iou_xyxy
from src.data.geometry import coord_bins_to_pixel_xyxy

B = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum")
OUTPUT = B / "second-fit-step64-introduced-owner-packets-v1"
SCORED = B / "second-fit-selectors-v1/scored-step-64.json"
READBACK = B / "second-fit-readback-recovery-v1/readback-step-64.json"
RAW_ROOT = B / "second-fit-readback-recovery-v1/rows"
OWNER_PLAN = B / "stage03-single-owner-repair-v1/owner-plan.json"
TARGET = B / "target-owners-complete-v3.json"
ACQUISITION = B / "stage01-acquisition-v1-retry1-config-batch2/manifest.json"
PATTERNS = B / "second-fit-stop-patterns-v1/per-output.jsonl"
PARSER_SOURCE = WORKTREE / "probes/source_rweak_row_cross/run.py"
SELECTOR_SOURCE = WORKTREE / "probes/training_set_completion/readback_selectors.py"
TOKENIZER_ROOT = Path("/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent")
IMAGE_IDS = (25274, 59571, 99937, 219546, 351017, 388795, 417044, 477415, 528944)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value))


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, box: list[int], colour: tuple[int, int, int], label: str, width: int = 4) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)), outline=colour, width=width)
    f = font(17)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=f)
    y = max(0, y1 - (bottom - top) - 5)
    draw.rectangle((x1, y, min(draw._image.width - 1, x1 + right - left + 6), y + bottom - top + 5), fill=colour)
    draw.text((x1 + 3, y + 2), label, fill=(255, 255, 255), font=f)


def crop_window(boxes: list[list[int]], width: int, height: int, *, scale: float, minimum: int) -> tuple[int, int, int, int]:
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    bw = max(minimum, x2 - x1); bh = max(minimum, y2 - y1)
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
    ww = min(width, bw * scale); hh = min(height, bh * scale)
    left = max(0, min(width - ww, cx - ww / 2)); top = max(0, min(height - hh, cy - hh / 2))
    return tuple(map(round, (left, top, left + ww, top + hh)))


def render_crop(image: Image.Image, gt: list[int], candidate: list[int], window: tuple[int, int, int, int], path: Path, label: str) -> None:
    view = image.crop(window)
    draw = ImageDraw.Draw(view)
    ox, oy = window[:2]
    gt_local = [gt[0]-ox, gt[1]-oy, gt[2]-ox, gt[3]-oy]
    cand_local = [candidate[0]-ox, candidate[1]-oy, candidate[2]-ox, candidate[3]-oy]
    # Keep a green rim visible when candidate and reference are exact matches.
    draw_box(draw, gt_local, (20, 170, 50), "selected GT", width=8)
    draw_box(draw, cand_local, (220, 45, 45), label, width=3)
    view.save(path, format="PNG", optimize=False)


def token_span(raw: dict[str, Any], offsets: list[tuple[int, int]], token_ids: list[int]) -> dict[str, Any]:
    start, end = int(raw["char_start"]), int(raw["char_end"])
    indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    require(indices and indices == list(range(indices[0], indices[-1] + 1)), "raw row does not map to contiguous tokens")
    return {
        "start_inclusive": indices[0], "end_exclusive": indices[-1] + 1,
        "token_ids": token_ids[indices[0]:indices[-1] + 1],
        "token_ids_sha256": sha256_bytes(canonical(token_ids[indices[0]:indices[-1] + 1])),
        "char_start": start, "char_end": end,
    }


def build() -> dict[str, Any]:
    require(OUTPUT.exists() and not (OUTPUT / "manifest.json").exists(), "output already finalized")
    scored = json.loads(SCORED.read_text()); readback = json.loads(READBACK.read_text())
    plan = json.loads(OWNER_PLAN.read_text()); target = json.loads(TARGET.read_text())
    acquisition = json.loads(ACQUISITION.read_text())
    patterns = [json.loads(line) for line in PATTERNS.read_text().splitlines() if line]
    plan_by_image = {int(row["image_id"]): row for row in plan["rows"] if row.get("action") == "append_absent_gt_before_eos_then_release"}
    target_by_key = {(int(row["image_id"]), str(row["owner_id"])): row for row in target["records"]}
    scored_by_image = {int(row["image_id"]): row for row in scored["rows"]}
    readback_by_image = {int(row["image_id"]): row for row in readback["rows"]}
    acquisition_by_image = {int(row["image_id"]): row for row in acquisition["records"]}
    pattern_by_image = {int(row["image_id"]): row for row in patterns if int(row["step"]) == 64}
    require(tuple(sorted(plan_by_image)) == tuple(sorted(IMAGE_IDS)), "append-owner plan cohort mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_ROOT), local_files_only=True)
    packets = []
    for image_id in IMAGE_IDS:
        selected = plan_by_image[image_id]
        target_row = target_by_key[(image_id, str(selected["owner_id"]))]
        require(target_row["reference_coord_bins_1000"] == selected["reference_bins"], f"target/plan geometry mismatch: {image_id}")
        require(target_row["category"] == selected["category"], f"target/plan class mismatch: {image_id}")
        score = scored_by_image[image_id]; saved = readback_by_image[image_id]
        raw_path = RAW_ROOT / f"step-00064-image-{image_id:012d}.json"
        raw_saved = json.loads(raw_path.read_text())
        for key in ("generated_token_ids", "generated_token_ids_sha256", "raw_decode_text", "decode_stop_reason", "prompt_token_ids", "route_id"):
            require(raw_saved[key] == saved[key], f"individual/consolidated readback mismatch {image_id}:{key}")
        tokenized = tokenizer(raw_saved["raw_decode_text"], add_special_tokens=False, return_offsets_mapping=True)
        require(tokenized["input_ids"] == raw_saved["generated_token_ids"], f"tokenizer replay mismatch: {image_id}")
        qualified = []
        all_valid = []
        for candidate in score["raw_rows"]:
            if candidate["status"] != "parsed_valid":
                continue
            overlap = iou_xyxy(selected["reference_bins"], candidate["coord_bins_1000"])
            enriched = dict(candidate); enriched["selected_reference_iou"] = overlap
            all_valid.append(enriched)
            if overlap >= 0.5:
                qualified.append(enriched)
        fallback = False
        if not qualified:
            require(all_valid, f"no parsed-valid candidate: {image_id}")
            qualified = [max(all_valid, key=lambda row: row["selected_reference_iou"])]
            fallback = True
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in qualified:
            signature = (row["description"], tuple(row["coord_bins_1000"]))
            groups.setdefault(signature, []).append(row)
        record = acquisition_by_image[image_id]
        image_path = Path(record["case"]["image_path"]).resolve(strict=True)
        with Image.open(image_path) as src:
            image = src.convert("RGB")
        require([image.width, image.height] == [record["case"]["image_width"], record["case"]["image_height"]], f"image dimensions mismatch: {image_id}")
        gt_pixel = list(coord_bins_to_pixel_xyxy(selected["reference_bins"], image_width=image.width, image_height=image.height, field="selected_gt"))
        packet_dir = OUTPUT / f"image-{image_id:012d}"
        packet_dir.mkdir()
        overlay = image.copy(); draw = ImageDraw.Draw(overlay)
        draw_box(draw, gt_pixel, (20, 170, 50), f"selected GT {selected['owner_id']} {selected['category']}", width=9)
        group_payloads = []
        for ordinal, ((description, bins_tuple), members) in enumerate(sorted(groups.items(), key=lambda item: min(int(x["generated_order"]) for x in item[1])), 1):
            bins = list(bins_tuple)
            candidate_pixel = list(coord_bins_to_pixel_xyxy(bins, image_width=image.width, image_height=image.height, field=f"candidate-{ordinal}"))
            overlap = iou_xyxy(selected["reference_bins"], bins)
            label = f"cand {ordinal} {description} {overlap:.3f}"
            draw_box(draw, candidate_pixel, (220, 45, 45), label, width=4)
            tight = packet_dir / f"candidate-{ordinal:02d}-tight.png"
            context = packet_dir / f"candidate-{ordinal:02d}-context.png"
            render_crop(image, gt_pixel, candidate_pixel, crop_window([gt_pixel, candidate_pixel], image.width, image.height, scale=1.25, minimum=56), tight, label)
            render_crop(image, gt_pixel, candidate_pixel, crop_window([gt_pixel, candidate_pixel], image.width, image.height, scale=3.0, minimum=160), context, label)
            member_payloads = []
            for member in sorted(members, key=lambda row: int(row["generated_order"])):
                raw = member["raw"]
                member_payloads.append({
                    "prediction_id": member["prediction_id"], "generated_order": int(member["generated_order"]),
                    "raw_span_sha256": member["raw_span_sha256"], "raw_span_text": raw["raw_span_text"],
                    "token_span": token_span(raw, tokenized["offset_mapping"], raw_saved["generated_token_ids"]),
                    "schema_spans": raw["schema_spans"], "coord_token_spans": raw["coord_token_spans"],
                })
            group_payloads.append({
                "candidate_group_id": f"image-{image_id:012d}:candidate-{ordinal:02d}",
                "selection": "iou_ge_0.5_to_selected_gt" if not fallback else "top_parsed_valid_fallback_no_match_claim",
                "description": description, "coord_bins_1000": bins, "bbox_pixel_xyxy": candidate_pixel,
                "selected_reference_iou": overlap, "member_count": len(member_payloads), "members": member_payloads,
                "shared_visual_evidence": {"tight_crop": None, "context_crop": None},
            })
            group_payloads[-1]["shared_visual_evidence"] = {"tight_crop": binding(tight), "context_crop": binding(context)}
        overlay_path = packet_dir / "selected-gt-and-candidates-overlay.png"
        overlay.save(overlay_path, format="PNG", optimize=False)
        pattern = pattern_by_image[image_id]
        context_fields = {key: pattern.get(key) for key in (
            "stop_reason", "capped", "natural_eos", "token_count", "native_parser",
            "first_observable_anomaly_generated_order", "sustained_repeat_or_collapse_generated_order",
            "eventual_exact_token_period", "longest_identical_row_run", "observed_failure_family", "prefix",
        )}
        packet = {
            "schema": "training_set_completion.second_fit_step64_introduced_owner_packet.v1",
            "status": "candidate_targeted_visual_packet_ready",
            "image_id": image_id, "step": 64,
            "review_question": "Does the selected physical GT owner appear with reasonable geometry in this candidate, regardless of whether its class text is correct?",
            "selected_gt_reference": {"owner_plan": selected, "target_v3_record": target_row, "bbox_pixel_xyxy": gt_pixel},
            "candidate_selection": {
                "rule": "every parsed-valid candidate with class-agnostic normalized-bin IoU >= 0.5 to this selected GT; if none, top parsed-valid candidate without a match claim",
                "parsed_valid_total": score["valid_prediction_count"], "selected_member_total": sum(len(v) for v in groups.values()),
                "selected_distinct_geometry_class_groups": len(groups), "fallback_used": fallback,
            },
            "candidate_groups": group_payloads,
            "visual_evidence": {"original_image": binding(image_path), "selected_gt_and_candidates_overlay": binding(overlay_path)},
            "full_raw_source": binding(raw_path),
            "raw_output_identity": {
                "route_id": raw_saved["route_id"], "checkpoint_step": raw_saved["checkpoint_step"],
                "checkpoint_adapter": raw_saved["checkpoint_adapter"], "model_receipt": raw_saved["model_receipt"],
                "generated_token_count": len(raw_saved["generated_token_ids"]),
                "generated_token_ids_sha256": raw_saved["generated_token_ids_sha256"],
                "raw_decode_text_sha256": sha256_bytes(raw_saved["raw_decode_text"].encode()),
                "prompt_token_ids_sha256": raw_saved["prompt_token_ids_sha256"],
                "executed_media_sha256": raw_saved["executed_media_sha256"],
                "decode_stop_reason": raw_saved["decode_stop_reason"],
            },
            "output_context_do_not_mistake_for_clean_output": context_fields,
            "source_bindings": {
                "scored_step64": binding(SCORED), "consolidated_readback_step64": binding(READBACK),
                "owner_plan": binding(OWNER_PLAN), "target_v3": binding(TARGET), "acquisition_manifest": binding(ACQUISITION),
                "stop_pattern_rows": binding(PATTERNS), "native_parser_source": binding(PARSER_SOURCE),
                "selector_source": binding(SELECTOR_SOURCE),
                "tokenizer_json": binding(TOKENIZER_ROOT / "tokenizer.json"),
            },
            "claim_boundary": "Viewing packet only. IoU is a candidate selector, not owner truth. Review only the selected GT and listed candidates; do not infer truth, false positives, cleanliness, or quality of other output rows.",
        }
        packet_path = packet_dir / "packet.json"; write_json(packet_path, packet)
        packets.append({
            "image_id": image_id, "selected_owner_id": str(selected["owner_id"]), "selected_category": selected["category"],
            "candidate_members": packet["candidate_selection"]["selected_member_total"],
            "candidate_groups": len(groups), "fallback_used": fallback,
            "capped": context_fields["capped"], "first_anomaly": context_fields["first_observable_anomaly_generated_order"],
            "sustained_repeat_or_collapse": context_fields["sustained_repeat_or_collapse_generated_order"],
            "packet": binding(packet_path),
        })
    manifest = {
        "schema": "training_set_completion.second_fit_step64_introduced_owner_packets.v1",
        "status": "candidate_targeted_visual_packets_ready", "step": 64,
        "image_ids": list(IMAGE_IDS), "image_count": len(packets), "packets": packets,
        "selection_summary": {"candidate_members": sum(x["candidate_members"] for x in packets), "candidate_groups": sum(x["candidate_groups"] for x in packets), "fallback_packets": sum(x["fallback_used"] for x in packets)},
        "builder": binding(Path(__file__)),
        "claim_boundary": "Nine independent selected-owner visual packets; no physical-owner, class, other-row, or output-cleanliness decision is made.",
    }
    write_json(OUTPUT / "manifest.json", manifest)
    return manifest


def validate() -> dict[str, Any]:
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "candidate_targeted_visual_packets_ready", "manifest status")
    require(manifest["image_ids"] == list(IMAGE_IDS) and manifest["image_count"] == 9, "manifest cohort")
    require(manifest["selection_summary"]["fallback_packets"] == 0, "unexpected fallback packet")
    for item in manifest["packets"]:
        packet_binding = item["packet"]; path = Path(packet_binding["path"])
        require(file_hash(path) == packet_binding["sha256"], f"packet hash mismatch: {path}")
        packet = json.loads(path.read_text())
        require(packet["image_id"] == item["image_id"] and packet["step"] == 64, "packet identity")
        require(not packet["candidate_selection"]["fallback_used"], "fallback selected")
        require(packet["candidate_selection"]["selected_member_total"] >= 1, "no selected candidate")
        for group in packet["candidate_groups"]:
            require(group["selected_reference_iou"] >= 0.5, "candidate below threshold")
            require(group["member_count"] == len(group["members"]), "member count mismatch")
            for member in group["members"]:
                start, end = member["token_span"]["start_inclusive"], member["token_span"]["end_exclusive"]
                require(0 <= start < end <= packet["raw_output_identity"]["generated_token_count"], "token span bounds")
            for evidence in group["shared_visual_evidence"].values():
                require(file_hash(Path(evidence["path"])) == evidence["sha256"], "crop hash mismatch")
        for evidence in packet["visual_evidence"].values():
            require(file_hash(Path(evidence["path"])) == evidence["sha256"], "visual hash mismatch")
        require(file_hash(Path(packet["full_raw_source"]["path"])) == packet["full_raw_source"]["sha256"], "raw source hash mismatch")
    return {"status": "validated", "manifest": binding(manifest_path), "image_count": 9, **manifest["selection_summary"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = validate() if args.validate_only else build()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
