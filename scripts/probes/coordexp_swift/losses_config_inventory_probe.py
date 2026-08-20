#!/usr/bin/env python
"""Model-free strict-resolution probe over the current supported config inventory.

Wave 1 of `standardize-coordexp-swift-supervised-losses` (task 1.5): resolve every
current supported CoordExp-Swift training config under the canonical roots, print
each path with its resolved config fingerprint and loss identity, and prove that a
representative historical/archived config is rejected by the current strict schema
(provenance, not a current input).

No model, tokenizer, GPU, or cache work happens here.

Usage:
    conda run -n ms python \
        scripts/probes/coordexp_swift/losses_config_inventory_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.errors import ConfigContractError  # noqa: E402
from src.config.loader import load_train_config  # noqa: E402


SUPPORTED_TRAIN_CONFIG_ROOTS = (
    Path("configs/coordexp_swift/prod"),
    Path("configs/coordexp_swift/smoke"),
)
HISTORICAL_CONFIGS = (
    Path(
        "configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/"
        "prod/compact_full_support2_instance_trie_focused_cap8_frac0p04_mix0p2.yaml"
    ),
)
CANONICAL_GROUPS = ("desc_text", "schema", "coordinate", "eos")
GATE_MODE_WEIGHTS = {"enabled": 0.1, "zero_weight_ablation": 0.0}


def _supported_config_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for root in SUPPORTED_TRAIN_CONFIG_ROOTS:
        if not root.is_dir():
            raise SystemExit(f"missing supported config root: {root}")
        paths.extend(sorted(root.rglob("*.yaml")))
    return tuple(paths)


def _loss_contract_failures(config_path: Path, config) -> list[str]:
    failures: list[str] = []
    losses = config.losses
    if losses.normalizer != "segment_balanced":
        failures.append(f"normalizer={losses.normalizer!r}")
    if losses.protected.base_ce.weight != 1.0:
        failures.append(f"base_ce.weight={losses.protected.base_ce.weight!r}")
    gate = losses.protected.token_type_gate
    if tuple(gate.groups) != CANONICAL_GROUPS:
        failures.append(f"token_type_gate.groups={list(gate.groups)!r}")
    if gate.mode not in GATE_MODE_WEIGHTS:
        failures.append(f"token_type_gate.mode={gate.mode!r}")
    elif gate.weight != GATE_MODE_WEIGHTS[gate.mode]:
        failures.append(f"token_type_gate {gate.mode} weight={gate.weight!r}")
    if hasattr(losses.protected, "coord_gaussian_rps"):
        failures.append("protected.coord_gaussian_rps still exists on the model")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="optional path for a machine-readable receipt",
    )
    args = parser.parse_args()

    failures: list[str] = []
    records: list[dict[str, object]] = []

    paths = _supported_config_paths()
    print(f"[inventory] supported training configs: {len(paths)}")
    for config_path in paths:
        try:
            resolved = load_train_config(config_path)
        except (ConfigContractError, ValueError) as exc:
            failures.append(f"{config_path}: strict resolution failed: {exc}")
            print(f"  FAIL  {config_path}: {exc}")
            continue
        contract_failures = _loss_contract_failures(config_path, resolved.config)
        gate = resolved.config.losses.protected.token_type_gate
        auxiliary = resolved.config.losses.auxiliary
        coord = auxiliary.coord_gaussian_rps if auxiliary is not None else None
        record = {
            "path": str(config_path),
            "fingerprint": resolved.fingerprint,
            "base_ce_weight": resolved.config.losses.protected.base_ce.weight,
            "token_type_gate_mode": gate.mode,
            "token_type_gate_weight": gate.weight,
            "token_type_gate_groups": list(gate.groups),
            "auxiliary_coord_gaussian_rps_weight": (
                None if coord is None else coord.weight
            ),
            "contract_failures": contract_failures,
        }
        records.append(record)
        status = "OK  " if not contract_failures else "FAIL"
        coord_note = "-" if coord is None else f"coord={coord.weight}"
        print(
            f"  {status}  {config_path}\n"
            f"        fingerprint={resolved.fingerprint}"
            f" gate={gate.mode}:{gate.weight} {coord_note}"
        )
        if contract_failures:
            failures.append(f"{config_path}: {'; '.join(contract_failures)}")

    print("[provenance] historical configs must be rejected by the current schema")
    historical: list[dict[str, object]] = []
    for config_path in HISTORICAL_CONFIGS:
        if not config_path.exists():
            failures.append(f"{config_path}: historical representative is missing")
            print(f"  FAIL  {config_path}: missing")
            continue
        try:
            load_train_config(config_path)
        except (ConfigContractError, ValueError) as exc:
            historical.append({"path": str(config_path), "rejection": str(exc)})
            print(f"  OK    {config_path} rejected: {type(exc).__name__}")
        else:
            failures.append(
                f"{config_path}: historical config was accepted by strict validation"
            )
            print(f"  FAIL  {config_path}: accepted")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(
                {
                    "supported_config_count": len(paths),
                    "supported_configs": records,
                    "historical_configs": historical,
                    "failures": failures,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if failures:
        print(f"[result] FAILED with {len(failures)} finding(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"[result] OK: {len(paths)} supported configs resolved strictly; "
        f"{len(historical)} historical config(s) rejected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
