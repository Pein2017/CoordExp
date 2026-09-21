"""CPU acceptance for saved visual-binding images, grids, and score contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .compositor import binding


def verify(root: Path) -> dict[str, Any]:
    admission = json.loads((root / "selection" / "shared-admission-binding.json").read_text())
    shared = json.loads(Path(admission["shared_admission"]["path"]).read_text())
    results = []
    for directory in sorted((root / "qualification-01").glob("*")):
        receipt_path, scores_path = directory / "receipt.json", directory / "scores.json"
        receipt, scores = json.loads(receipt_path.read_text()), json.loads(scores_path.read_text())
        if receipt["scores"] != binding(scores_path) or receipt["status"] != "candidate_complete":
            raise ValueError(f"state receipt is not complete: {directory.name}")
        state = next(item for item in shared["lane_b"]["states"] if item["id"] == scores["state_id"])
        target = next(item for item in shared["source_pool"] if item["source_boundary_id"] == scores["source_boundary_id"])
        source_path = Path(target["image"]["path"])
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
            source_bytes = source.tobytes()
            source_size = list(source.size)
        clean_path = Path(scores["compositor"]["clean"]["image"]["path"])
        with Image.open(clean_path) as opened:
            clean = opened.convert("RGB")
            if list(clean.size) != source_size or clean.tobytes() != source_bytes:
                raise ValueError(f"clean decoded RGB changed: {directory.name}")
        image_checks = []
        for name in ("A", "N", "unrelated"):
            condition = f"ablate_{name}"
            comp = scores["compositor"][condition]
            mask = state["masks"][name]
            x1, y1, x2, y2 = (int(value) for value in mask["pixel_xyxy"])
            if comp["decoded_size"] != source_size or comp["pixel_xyxy"] != [x1, y1, x2, y2]:
                raise ValueError(f"mask geometry changed: {directory.name}/{name}")
            expected_area = (x2 - x1) * (y2 - y1)
            if comp["changed_pixel_count"] != expected_area or comp["unchanged_complement_pixel_count"] != source_size[0] * source_size[1] - expected_area:
                raise ValueError(f"mask complement accounting changed: {directory.name}/{name}")
            image_checks.append({"region": name, "image": comp["image"], "mask": comp["mask"], "changed_pixel_count": comp["changed_pixel_count"], "unchanged_complement_pixel_count": comp["unchanged_complement_pixel_count"]})
        target_index = int(target["batch_index"])
        clean_grid = scores["conditions"]["clean"]["input_identity"]["image_grids"][target_index]
        if clean_grid != target["image_plan"]["observed_image_grid_thw"]:
            raise ValueError(f"clean image grid changed: {directory.name}")
        results.append({"state_id": scores["state_id"], "source_image": binding(source_path), "clean_decoded_rgb_sha256": hashlib.sha256(source_bytes).hexdigest(), "image_checks": image_checks, "clean_grid": clean_grid, "candidate_count": sum(len(group["row_ids"]) for group in scores["candidate_sets"].values()), "conditions": list(scores["conditions"])})
    audit = []
    for receipt_path in sorted((root / "states").glob("*/receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        audit.append({"state_id": receipt.get("state_id"), "receipt": binding(receipt_path), "disposition": "redundant_pre_ruling_execution_excluded_from_primary_cpu_acceptance"})
    result = {"schema": "visual_instance_binding.cpu_acceptance.v1", "status": "passed", "primary_campaign": "qualification-01", "states": results, "audit_only_redundant_execution": audit, "producer": binding(Path(__file__)), "reducer": binding(Path(__file__).with_name("reduce.py")), "command": "python -m probes.training_set_completion.visual_instance_binding.verify --root <root>"}
    out = root / "cpu-acceptance.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["binding"] = binding(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps({"status": result["status"], "states": len(result["states"]), "output": str(args.root / "cpu-acceptance.json")}, indent=2))


if __name__ == "__main__":
    main()
