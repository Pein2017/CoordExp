from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.autoreg_supervision_credit import run_supervision_credit


DEFAULT_ANALYSIS_ROOT = Path(
    "/data/CoordExp/outputs/analysis/autoreg_object_rollout/ckpt3664_val200"
)
DEFAULT_TRAINING_RUN_DIR = Path(
    "/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/"
    "compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/"
    "compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/"
    "v0-20260504-071356"
)
DEFAULT_CHECKPOINT = DEFAULT_TRAINING_RUN_DIR / "checkpoint-3664"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CPU-only Lane F supervision-credit evidence ledger.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help="Autoregressive rollout analysis root.",
    )
    parser.add_argument(
        "--training-run-dir",
        type=Path,
        default=DEFAULT_TRAINING_RUN_DIR,
        help="Stage-1 training run directory containing logging.jsonl and resolved_config.json.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint directory used to read trainer_state.json.",
    )
    parser.add_argument(
        "--dataset-slice",
        default=None,
        help="Optional override for the ledger scope dataset_slice.",
    )
    parser.add_argument(
        "--metric-family",
        default=None,
        help="Optional override for the ledger scope metric_family.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_supervision_credit(
        analysis_root=args.analysis_root,
        training_run_dir=args.training_run_dir,
        checkpoint=args.checkpoint,
        dataset_slice=args.dataset_slice,
        metric_family=args.metric_family,
    )
    output_dir = Path(args.analysis_root).resolve() / "supervision_credit"
    print(
        json.dumps(
            {
                "summary_json": str(output_dir / "summary.json"),
                "report_md": str(output_dir / "report.md"),
                "h5_status": summary["h5_readout"]["status"],
                "production_training_recommendation": summary["h5_readout"][
                    "production_training_recommendation"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
