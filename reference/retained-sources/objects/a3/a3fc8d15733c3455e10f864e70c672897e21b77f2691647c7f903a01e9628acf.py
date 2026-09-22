"""CPU-only acceptance checks for the frozen Lane-A package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path("/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-19-recurrence-distribution-census")


def binding(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), "kind": "file"}


def main() -> None:
    manifest = json.loads((OUT / "eligible-manifest.json").read_text())
    assert manifest["status"] == "frozen_before_new_outputs"
    assert manifest["source_records_seen"] == 122218
    assert manifest["eligible_unique_records"] == 121693
    assert manifest["selected_count"] == 128
    audit = manifest["identity_audit"]
    assert audit["status"] == "no_conflict"
    assert audit["prior_alias_count"] == audit["processed_alias_count"] == 0
    assert audit["prior_conflict_count"] == audit["processed_conflict_count"] == 0

    panel = json.loads((OUT / "shared-panel.json").read_text())
    assert panel["status"] == "final_frozen"
    assert panel["counts"] == {"existing": 25, "new": 20, "all": 45, "failure": 21, "nonrecurrent_proxy": 24}
    assert len(panel["all_boundaries"]) == 45
    assert len({x["id"] for x in panel["all_boundaries"]}) == 45
    assert all(x.get("split") for x in panel["all_boundaries"])
    assert all(Path(x["raw_path"]).exists() and Path(x["trace_path"]).exists() and Path(x["receipt_path"]).exists() for x in panel["all_boundaries"])
    selector = panel["selection_rule"]["selector_source"]
    assert binding(Path(selector["path"])) == selector

    new = json.loads((OUT / "new-census.json").read_text())
    assert new["status"] == "candidate_cpu_reduced"
    assert new["denominator"] == {"unique_images": 128, "outputs": 256, "conditions": ["tied-original", "untied-original"]}
    assert new["failed_groups"] == []
    assert all(len(v) == 128 for v in new["cells"].values())
    mature = json.loads((OUT / "mature-census.json").read_text())
    assert mature["status"] == "candidate_cpu_reduced"
    assert mature["denominator"]["unique_images"] == 145
    assert mature["denominator"]["outputs"] == 580
    for condition in ("tied-original", "untied-original"):
        receipts = [json.loads(p.read_text()) for p in sorted((OUT / "runtime" / condition).glob("new-*/receipt.json"))]
        assert len(receipts) == 32 and all(r["status"] == "candidate_complete" for r in receipts)
    print(json.dumps({"status": "pass", "eligible": manifest["eligible_unique_records"], "new_outputs": new["denominator"]["outputs"], "panel": panel["counts"]}, indent=2))


if __name__ == "__main__":
    main()
