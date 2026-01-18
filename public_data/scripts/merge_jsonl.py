"""Offline JSONL merge via a fusion config.

This is optional: CoordExp can train directly from `custom.fusion_config`.
However, materializing a fused JSONL can be useful for external tooling or for
creating a fixed dataset snapshot.

Usage:
  PYTHONPATH=. /root/miniconda3/envs/ms/bin/python public_data/scripts/merge_jsonl.py \\
    --fusion-config configs/fusion/examples/lvis_vg.yaml \\
    --output-jsonl output/fused.jsonl
"""

from __future__ import annotations

import argparse

from src.datasets.fusion import FusionConfig, build_fused_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge JSONLs according to a fusion config.")
    parser.add_argument("--fusion-config", required=True, help="Path to fusion YAML/JSON.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=2025, help="Shuffle/upsample seed.")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable final shuffle of the merged records.",
    )
    args = parser.parse_args()

    cfg = FusionConfig.from_file(args.fusion_config)
    out = build_fused_jsonl(
        cfg,
        args.output_jsonl,
        seed=int(args.seed),
        shuffle=not bool(args.no_shuffle),
    )
    print(str(out))


if __name__ == "__main__":
    main()

