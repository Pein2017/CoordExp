"""Production-facing BF16 vLLM qualification and admission entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

from src.inference.vllm_qualification_producer import (
    DEFAULT_CHILD_TIMEOUT_SECONDS,
    admit,
    produce,
)


def _positive_finite_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.qualify_vllm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="run isolated BF16 qualification children into an absent external root",
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument(
        "--child-timeout-seconds",
        type=_positive_finite_seconds,
        default=DEFAULT_CHILD_TIMEOUT_SECONDS,
        help="finite per-child execution deadline (default: 1800 seconds)",
    )

    admission = subparsers.add_parser(
        "admit",
        help="validate a complete receipt set and atomically admit it",
    )
    admission.add_argument("--config", type=Path, required=True)
    admission.add_argument("--receipts-root", type=Path, required=True)
    admission.add_argument("--target-root", type=Path)

    return parser


def _child_parser() -> argparse.ArgumentParser:
    child = argparse.ArgumentParser(prog="python -m src.qualify_vllm _child")
    child.add_argument(
        "--kind",
        choices=("composition", "runtime", "concurrency", "forced_replay"),
        required=True,
    )
    child.add_argument("--max-num-seqs", type=int, choices=(1, 4), required=True)
    child.add_argument("--config", type=Path, required=True)
    child.add_argument("--evidence-dir", type=Path, required=True)
    child.add_argument("--result", type=Path, required=True)
    return child


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values[:1] == ["_child"]:
        args = _child_parser().parse_args(values[1:])
        args.command = "_child"
    else:
        args = _parser().parse_args(values)
    if args.command == "run":
        result = produce(
            config_path=args.config,
            output_root=args.output_root,
            child_timeout_seconds=args.child_timeout_seconds,
        )
    elif args.command == "admit":
        kwargs = {
            "config_path": args.config,
            "receipts_root": args.receipts_root,
        }
        if args.target_root is not None:
            kwargs["target_root"] = args.target_root
        result = admit(**kwargs)
    else:
        from src.inference.vllm_qualification_runtime import run_child

        result = run_child(
            kind=args.kind,
            max_num_seqs=args.max_num_seqs,
            config_path=args.config,
            evidence_dir=args.evidence_dir,
        )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
