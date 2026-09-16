"""CPU preparation entry for the Source256 B-normalized successor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from probes.training_set_completion import source256_normalized_training as runtime
from probes.training_set_completion import training


ROOT = runtime.ROOT
SOURCE_CONFIG = runtime.SOURCE_CONFIG
SCHEMA = runtime.TRIAL_SCHEMA
PREPARATION = runtime.SHARED_PREPARATION
ARM = runtime.B_NORMALIZED


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def build_training_manifest(
    *,
    preparation_path: Path = PREPARATION,
    arm: str = ARM,
    mode: str = "qualification",
    output: Path,
) -> dict[str, Any]:
    require(arm == ARM, "normalized successor has one B-normalized arm")
    return runtime.build_training_manifest(
        preparation_path=preparation_path,
        mode=mode,
        output=output,
    )


def prepare(
    *,
    preparation_path: Path = PREPARATION,
    output: Path,
    mode: str = "qualification",
) -> dict[str, Any]:
    return runtime.prepare(
        preparation_path=preparation_path,
        output=output,
        mode=mode,
    )


def validate_trial(value: dict[str, Any], *, verify_sources: bool = True) -> dict[str, Any]:
    """Validate the trial pointer and arm binding without launching work."""

    required = {
        "schema",
        "status",
        "unit_id",
        "mode",
        "preparation",
        "preparation_mirror",
        "preparation_receipt",
        "accounting",
        "arm",
        "arms",
        "manifest",
        "qualification_terminal",
        "topology",
        "launch",
        "release_contract",
        "producer",
    }
    require(set(value) == required, "normalized trial fields")
    require(value["schema"] == SCHEMA, "normalized trial schema")
    require(value["unit_id"] == runtime.UNIT_ID, "normalized trial unit")
    require(value["arm"] == ARM and value["mode"] in ("qualification", "main"), "normalized trial arm/mode")
    require(
        value["arms"] == {ARM: value["manifest"]},
        "normalized trial arm manifest binding",
    )
    require(value["preparation"] == runtime.training.binding(PREPARATION), "shared preparation binding")
    require(value["preparation_mirror"]["sha256"] == runtime.SHARED_PREPARATION_SHA256, "preparation mirror digest")
    require(value["qualification_terminal"]["status"] == "pending_lead_release", "qualification held status")
    require(value["release_contract"]["required_arm"] == ARM, "release arm")
    if verify_sources:
        require(training.binding(value["producer"]["path"]) == value["producer"], "trial producer changed")
        require(training.binding(value["preparation_receipt"]["path"]) == value["preparation_receipt"], "preparation receipt changed")
        require(training.binding(value["manifest"]["path"]) == value["manifest"], "training manifest changed")
    runtime.validate_training_manifest(read(value["manifest"]["path"]))
    return dict(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="CPU-only preparation and manifest")
    prepare_parser.add_argument("--preparation", type=Path, default=PREPARATION)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--mode", choices=("qualification", "main"), default="qualification")
    verify_parser = sub.add_parser("verify", help="CPU-only trial validation")
    verify_parser.add_argument("--trial", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(preparation_path=args.preparation, output=args.output, mode=args.mode), sort_keys=True))
    else:
        validate_trial(read(args.trial))


if __name__ == "__main__":
    main()
