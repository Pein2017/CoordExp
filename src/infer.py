"""Thin public CoordExp inference entrypoint."""

from __future__ import annotations

import argparse

from src.inference import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.infer")
    parser.add_argument("--config", required=True, help="CoordExp inference YAML")
    args = parser.parse_args(argv)
    return pipeline.run(config_path=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
