#!/usr/bin/env python3
"""Provision one resident Label Studio ROI profile and launch config."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.label_studio_coco_refinement.inference_profiles import canonical_json
from src.label_studio_coco_refinement.profile_provisioning import (
    provision_roi_profile,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", required=True)
    parser.add_argument("--profile-store", required=True)
    parser.add_argument("--launch-config", required=True)
    parser.add_argument("--receipt-store", required=True)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--bind-port", required=True, type=int)
    parser.add_argument("--ack-timeout-seconds", required=True, type=float)
    parser.add_argument("--default-width", type=int, default=1024)
    parser.add_argument("--default-height", type=int, default=1024)
    parser.add_argument("--min-axis-pixels", required=True, type=int)
    parser.add_argument("--max-axis-pixels", required=True, type=int)
    parser.add_argument("--max-total-pixels", required=True, type=int)
    parser.add_argument("--deadline-seconds", required=True, type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = provision_roi_profile(
        infer_config_path=args.infer_config,
        profile_store_path=args.profile_store,
        launch_config_path=args.launch_config,
        receipt_store_path=args.receipt_store,
        profile_name=args.profile_name,
        selector=args.selector,
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        insertion_ack_timeout_seconds=args.ack_timeout_seconds,
        default_width=args.default_width,
        default_height=args.default_height,
        min_axis_pixels=args.min_axis_pixels,
        max_axis_pixels=args.max_axis_pixels,
        max_total_pixels=args.max_total_pixels,
        deadline_seconds=args.deadline_seconds,
    )
    print(canonical_json(receipt), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
