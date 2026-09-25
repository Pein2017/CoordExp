"""Eight-rank, three-loss first-entry checks for ordinary adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path

from probes.coordinate_representation.coordinate_codebook_alignment import scale_train as scale
from probes.coordinate_representation.coordinate_codebook_alignment.early_edge_train import _patched_trainer


def run(config: Path, output: Path, packing_plan: Path):
    old = (
        scale.EXPECTED_WORLD_SIZE,
        scale.EXPECTED_GRAD_ACCUM,
        scale.EXPECTED_UPDATES,
        scale.EXPECTED_CATEGORIES,
        scale._patch_trainer,
    )
    scale.EXPECTED_WORLD_SIZE = 8
    scale.EXPECTED_GRAD_ACCUM = 1
    scale.EXPECTED_UPDATES = 984
    scale.EXPECTED_CATEGORIES = ("language", "vision", "aligner", "input", "output")
    scale._patch_trainer = lambda probe: _patched_trainer(
        probe, output.with_name("three-loss-objective.json")
    )
    try:
        return scale.run(config, output, packing_plan)
    finally:
        (
            scale.EXPECTED_WORLD_SIZE,
            scale.EXPECTED_GRAD_ACCUM,
            scale.EXPECTED_UPDATES,
            scale.EXPECTED_CATEGORIES,
            scale._patch_trainer,
        ) = old


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packing-plan", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.packing_plan)
