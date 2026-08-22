#!/usr/bin/env python3
"""Build the flat COCO length-budget training view from a prepared source preset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from public_data.scripts.build_coco_views import estimate_total_tokens
from public_data.scripts.convert_to_coord_tokens import convert_record_to_ints, convert_record_to_tokens


def build_length_budget_artifacts(
    *, source_preset: Path, output_root: Path, splits: Sequence[str], max_total_tokens: int,
) -> dict[str, int]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output target is not fresh: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in splits:
        source = source_preset / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"missing source split: {source}")
        outputs = {
            "raw": output_root / f"{split}.jsonl",
            "norm": output_root / f"{split}.norm.jsonl",
            "coord": output_root / f"{split}.coord.jsonl",
        }
        handles = {name: path.open("w", encoding="utf-8") for name, path in outputs.items()}
        accepted = 0
        rejected = 0
        try:
            with source.open("r", encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    raw: dict[str, Any] = json.loads(line)
                    normalized = convert_record_to_ints(
                        raw, ("bbox_2d", "poly", "line"), assume_normalized=False
                    )
                    if estimate_total_tokens(normalized) > max_total_tokens:
                        rejected += 1
                        continue
                    coord = convert_record_to_tokens(normalized, ("bbox_2d", "poly", "line"))
                    handles["raw"].write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handles["norm"].write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handles["coord"].write(json.dumps(coord, ensure_ascii=False, separators=(",", ":")) + "\n")
                    accepted += 1
        finally:
            for handle in handles.values():
                handle.close()
        if accepted == 0:
            raise RuntimeError(f"length-budget output is empty for split {split!r}")
        (output_root / f"{split}.length_budget_stats.json").write_text(
            json.dumps({"accepted": accepted, "rejected": rejected, "max_total_tokens": max_total_tokens}, indent=2) + "\n",
            encoding="utf-8",
        )
        counts[split] = accepted
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max-total-tokens", type=int, default=12000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    counts = build_length_budget_artifacts(**vars(parse_args(argv)))
    for split, count in counts.items():
        print(f"[length-budget] {split}: {count} records")


if __name__ == "__main__":
    main()
