"""First-two-update checks for the fixed four-rank paired 2048 fits."""

from __future__ import annotations

import argparse
from pathlib import Path

from probes.coordinate_representation.coordinate_codebook_alignment import scale_train as scale
from probes.coordinate_representation.coordinate_codebook_alignment import three_loss_train


def run(arm: str, config: Path, output: Path, packing_plan: Path, *, smoke: bool = False, order_gate: bool = False):
    if arm not in {"early", "late"}:
        raise ValueError(f"unsupported paired arm: {arm}")
    old = (scale.EXPECTED_UPDATES, scale._patch_trainer)
    scale.EXPECTED_UPDATES = 2 if smoke else 472
    expected_weights = (
        {"base_ce": 1.0, "token_type_gate": 0.2, "conditional_order_gate": 0.2}
        if order_gate else None
    )
    scale._patch_trainer = lambda probe: three_loss_train.patched(
        probe, output.with_name(f"{arm}-three-loss-objective.json"),
        **({"expected_weights": expected_weights} if order_gate else {}),
    )
    try:
        return scale.run(config, output, packing_plan)
    finally:
        scale.EXPECTED_UPDATES, scale._patch_trainer = old


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("early", "late"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packing-plan", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--order-gate", action="store_true")
    args = parser.parse_args()
    run(args.arm, args.config, args.output, args.packing_plan, smoke=args.smoke, order_gate=args.order_gate)
