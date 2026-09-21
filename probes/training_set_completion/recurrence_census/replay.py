"""Replay the saved Lane-A raw outputs into a separate CPU reduction file.

This command is deliberately read-only with respect to the frozen panel and
source records.  It calls the existing cell scorer and aggregate reducer, but
never runs panel selection and requires an output path separate from the
source root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from probes.training_set_completion.recurrence_census import reduce_new
from probes.training_set_completion import row_scoring
from probes.training_set_completion.recurrence_census.prepare import binding


def _reduce(source_root: Path) -> dict[str, Any]:
    panel = json.loads((source_root / "panel.json").read_text())
    # _cell resolves saved raw/trace/receipt bindings through reduce_new.OUT.
    # Rebinding this module global keeps the existing scorer pure and avoids a
    # second implementation of the saved-output semantics.
    reduce_new.OUT = source_root
    score_mod = row_scoring
    cells: dict[str, dict[str, Any]] = {}
    failed: list[dict[str, Any]] = []
    for condition in panel["conditions"]:
        by_image: dict[str, dict[str, Any]] = {}
        for group in panel["groups"]:
            runtime = source_root / "runtime" / condition / group["key"]
            receipt_path = runtime / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "candidate_complete":
                failed.append({"condition": condition, "group": group["key"], "status": receipt.get("status")})
                continue
            raw = json.loads((runtime / "raw.json").read_text())
            for j, case in enumerate(group["cases"]):
                key = f"{case['input_record']['metadata']['split']}:{case['input_record']['image_id']}"
                bank = panel["new_banks"][key]
                by_image[key] = reduce_new._cell(condition, group, j, raw, case, score_mod, bank)
        cells[condition] = by_image
    if failed:
        raise RuntimeError(f"saved output groups are not complete: {failed}")

    summary = {condition: reduce_new._aggregate(list(values.values())) for condition, values in cells.items()}
    n_images = len({key for values in cells.values() for key in values})
    return {
        "schema": "recurrence_census.replayed_new_census.v1",
        "status": "candidate_cpu_replayed",
        "source_panel": binding(source_root / "panel.json"),
        "source_raw_runtime": binding(source_root / "runtime"),
        "denominator": {
            "unique_images": n_images,
            "outputs": sum(len(values) for values in cells.values()),
            "conditions": panel["conditions"],
        },
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if output == source_root / "shared-panel.json" or output == source_root / "shared-sources.json":
        raise SystemExit("refusing to overwrite a frozen shared artifact")
    result = _reduce(source_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "denominator": result["denominator"]}, indent=2))


if __name__ == "__main__":
    main()
