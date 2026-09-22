"""Build the immutable 128-image natural-output panel after CPU selection."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from probes.training_set_completion.recurrence_census.prepare import OUT, MATURE, binding, COORD


def main() -> None:
    runtime = OUT / "new128.runtime.jsonl"
    rows = [json.loads(line) for line in runtime.read_text().splitlines() if line.strip()]
    numeric_ids = [int(row["image_id"]) for row in rows]
    if len(numeric_ids) != len(set(numeric_ids)):
        raise RuntimeError("selected cohort reuses a numeric image_id across splits; consumer bank keys must remain split-qualified")
    mature = json.loads((MATURE / "panel.json").read_text())
    configs = copy.deepcopy(mature["configs"])
    groups = []
    banks = {}
    for i, row in enumerate(rows):
        split = str(row["metadata"]["split"])
        image_id = int(row["image_id"])
        iid = str(image_id)
        row_id = f"coco2017_{split}_{image_id:012d}"
        case = {
            "row_id": row_id,
            "row_index": i,
            "image_id": image_id,
            "image_width": int(row["width"]),
            "image_height": int(row["height"]),
            "input_record": row,
        }
        bank = []
        for j, obj in enumerate(row.get("objects", [])):
            bins = [int(COORD.fullmatch(v).group(1)) for v in obj["bbox_2d"]]
            desc = str(obj.get("desc", "")).strip()
            bank.append(
                {
                    "image_id": image_id,
                    "owner_id": str(obj.get("coco_ann_id", f"{image_id}:object:{j}")),
                    "reference_coord_bins_1000": bins,
                    "description": desc,
                    "normalized_description": desc.lower(),
                    "class_status": "verified_coco80",
                }
            )
        banks[f"{split}:{image_id}"] = bank
        if i % 4 == 0:
            groups.append({
                "key": f"new-{i // 4:02d}",
                "cohort": "prospective_seed19",
                "input_jsonl": str(runtime),
                "cases": [],
            })
        groups[-1]["cases"].append(case)
    for config in configs.values():
        config["data"] = {"input_jsonl": str(runtime)}
        config["generation"] = {
            **config.get("generation", {}),
            "batch_size": 4,
            "max_new_tokens": 3084,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "repetition_penalty": 1.0,
        }
        config["run"] = {**config.get("run", {}), "artifact_root": str(OUT), "name": "recurrence-census-original-natural"}
    panel = {
        "schema": "recurrence_census.natural_panel.v1",
        "status": "ready_for_real_entry_qualification",
        "unit_id": "2026-09-19-recurrence-distribution-census",
        "conditions": ["tied-original", "untied-original"],
        "configs": configs,
        "groups": groups,
        "input_jsonl": str(runtime),
        "new_banks": banks,
        "sources": {
            "runtime": binding(runtime),
            "eligible_manifest": binding(OUT / "eligible-manifest.json"),
            "prelaunch_panel": binding(OUT / "prelaunch-panel.json"),
            "shared_sources": binding(OUT / "shared-sources.json"),
        },
        "generation_contract": {
            "readout": "original",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "rp": "RP1",
            "native_cap_tokens": 3084,
            "outputs": 256,
            "qualification": "first group per model uses actual native forward/readback before remaining groups",
        },
    }
    (OUT / "panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    print(json.dumps({"status": panel["status"], "groups": len(groups), "cases": len(rows), "outputs": len(rows) * 2, "panel": str(OUT / "panel.json")}, indent=2))


if __name__ == "__main__":
    main()
