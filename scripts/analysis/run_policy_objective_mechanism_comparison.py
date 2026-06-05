from __future__ import annotations

import argparse
import json

from src.analysis.policy_objective_mechanism_comparison.config import load_config
from src.analysis.policy_objective_mechanism_comparison.runner import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        payload = {
            "status": "dry_run_ok",
            "artifact_root": str(config.artifact_root),
            "checkpoint_roles": list(config.checkpoint_roles),
            "artifact_sources": {
                name: str(artifact.root)
                for name, artifact in config.artifact_sources.items()
            },
        }
    else:
        payload = run(config, config_path=args.config, allow_overwrite=True)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
