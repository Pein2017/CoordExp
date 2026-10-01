"""Read-only aggregation of the fixed 32-image COCO/LVIS visual-review protocol.

No annotation export, image selection, model loading, or input-contract mutation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def aggregate(root: Path) -> dict[str, Any]:
    root = Path(root)
    contract = json.loads((root / "contract.json").read_text())
    sample_path = root / "sample.jsonl"
    actual = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    _require(actual == contract["sample_sha256"], "sample SHA-256 mismatch")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(sample_path):
        groups[row["source"]].append(row)
    _require(bool(groups), "empty review sample")
    counts, label_hashes = {}, {}
    for source, sample in groups.items():
        _require(Path(source).name == source and source not in (".", ".."), "invalid source name")
        path = root / f"{source}.labels.jsonl"
        labels = _rows(path)
        keys = lambda rows: {(x["index"], x["image_id"]) for x in rows}
        _require(len(labels) == len(sample) == 32, f"{source}: expected exactly 32 labels and samples")
        _require(keys(labels) == keys(sample) and len({x["image_id"] for x in labels}) == 32,
                 f"{source}: label/sample identity mismatch")
        by_id = {x["image_id"]: x for x in sample}
        c, split, target = Counter(), defaultdict(Counter), defaultdict(Counter)
        for x in labels:
            _require(x["source"] == source and x["presence"] in ("yes", "no", "uncertain"), "invalid presence/source")
            _require(x["box_proxy"] in ("yes", "no", "uncertain", "not_applicable"), "invalid box proxy")
            _require(type(x["embedded"]) is bool, "embedded must be boolean")
            _require(x["presence"] == "yes" or x["box_proxy"] == "not_applicable", "nonpositive presence cannot localize target")
            _require(not (x["presence"] in ("no", "uncertain") or x["embedded"]) or bool(x["note"]), "missing review note")
            row = by_id[x["image_id"]]
            c[x["presence"]] += 1
            c["embedded_yes"] += x["embedded"] and x["presence"] == "yes"
            c["box_proxy_yes"] += x["box_proxy"] == "yes"
            split[row["coco_split"]][x["presence"]] += 1
            target[row["target_lvis_status"]][x["presence"]] += 1
        counts[source] = dict(presence=dict(c), by_coco_split={k: dict(v) for k, v in split.items()},
                              by_lvis_target_status={k: dict(v) for k, v in target.items()}, population=contract["population"][source])
        label_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = root / "tablecloth.inference.jsonl"
    inference = _rows(path)
    _require("tablecloth" in groups and len(inference) == 32 and keys(inference) == keys(groups["tablecloth"]),
             "scene-inference identity mismatch")
    _require(all(x["likely_dining_table"] in ("high", "medium", "low") for x in inference), "invalid inference level")
    counts["tablecloth"]["scene_inference"] = dict(Counter(x["likely_dining_table"] for x in inference))
    label_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(status="exploratory_review_complete", sample_sha256=actual, label_sha256=label_hashes, counts=counts,
                limits=["32 fixed image-level samples per relation, split-stratified rather than population proportional",
                        "visual presence does not establish original COCO annotation error",
                        "one reviewer per relation; uncertain labels retained",
                        "no dataset export or training in this round"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Frozen review directory")
    parser.add_argument("--output", required=True, type=Path, help="Absent result JSON in the owner worktree")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; no overwrite is permitted")
    try:
        result = aggregate(args.input)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
