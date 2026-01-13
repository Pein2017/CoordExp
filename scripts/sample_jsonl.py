#!/root/miniconda3/envs/ms/bin/python
"""
Uniformly sample N lines from a JSONL file (without replacement).

Uses reservoir sampling so it works for large files without loading everything into memory.

Example:
  /root/miniconda3/envs/ms/bin/python scripts/sample_jsonl.py \
    --input public_data/lvis/rescale_32_768_poly_20/train.coord.jsonl \
    --output public_data/lvis/rescale_32_768_poly_20/train_5k.coord.jsonl \
    --num-samples 5000 --seed 42 --write-meta
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleMeta:
    input_path: str
    output_path: str
    num_samples: int
    seed: int
    total_nonempty_lines_seen: int
    method: str = "reservoir"
    created_at_unix: int = 0


def _reservoir_sample_lines(input_path: Path, num_samples: int, seed: int) -> tuple[list[str], int]:
    rng = random.Random(seed)

    reservoir: list[str] = []
    total_nonempty = 0

    with input_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip("\n")
            if not line.strip():
                continue
            total_nonempty += 1

            if len(reservoir) < num_samples:
                reservoir.append(line)
                continue

            # Replace elements with gradually decreasing probability.
            j = rng.randrange(total_nonempty)
            if j < num_samples:
                reservoir[j] = line

    if total_nonempty < num_samples:
        raise ValueError(
            f"Not enough non-empty lines to sample: requested {num_samples}, found {total_nonempty} in {input_path}"
        )

    rng.shuffle(reservoir)
    return reservoir, total_nonempty


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample N lines from a JSONL file (without replacement).")
    parser.add_argument("--input", required=True, type=str, help="Input .jsonl path")
    parser.add_argument("--output", required=True, type=str, help="Output .jsonl path")
    parser.add_argument("--num-samples", required=True, type=int, help="Number of lines to sample")
    parser.add_argument("--seed", default=42, type=int, help="RNG seed (default: 42)")
    parser.add_argument(
        "--write-meta",
        action="store_true",
        help="Write sidecar meta JSON to <output>.meta.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    num_samples = int(args.num_samples)
    seed = int(args.seed)

    if num_samples <= 0:
        raise ValueError("--num-samples must be > 0")

    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples, total_nonempty = _reservoir_sample_lines(input_path, num_samples=num_samples, seed=seed)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out_f:
        for line in samples:
            out_f.write(line)
            out_f.write("\n")
    os.replace(tmp_path, output_path)

    if args.write_meta:
        meta = SampleMeta(
            input_path=str(input_path),
            output_path=str(output_path),
            num_samples=num_samples,
            seed=seed,
            total_nonempty_lines_seen=total_nonempty,
            created_at_unix=int(time.time()),
        )
        meta_path = Path(str(output_path) + ".meta.json")
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        with tmp_meta.open("w", encoding="utf-8") as meta_f:
            json.dump(asdict(meta), meta_f, ensure_ascii=True, indent=2, sort_keys=True)
            meta_f.write("\n")
        os.replace(tmp_meta, meta_path)

    print(f"Wrote {num_samples} samples to: {output_path}")
    print(f"Total non-empty lines seen: {total_nonempty}")
    if args.write_meta:
        print(f"Wrote meta: {str(output_path)}.meta.json")


if __name__ == "__main__":
    main()

