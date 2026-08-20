#!/usr/bin/env python
"""Model-free pack-cache determinant equality probe (task 5.3 gate).

Wave 5 of `standardize-coordexp-swift-supervised-losses`. The change's stop
rule is that loss config and objective code are NOT pack-cache semantic
owners: after every source and supported-config edit, the train and
`eval.forward` determinant payloads and aggregate fingerprints MUST still be
byte-identical to the Wave-0 entry baseline frozen in
`receipts/wave-0-determinant-baseline.json`. Any difference is a blocking
contract failure, never a rebuild.

Scope and guards:

- The baseline records determinants for exactly ONE config (entry-audit F-10),
  so this probe pins that path as a constant and refuses to run when the
  baseline's own `config` field disagrees. A config substitution therefore
  cannot silently satisfy the invariant.
- The baseline document authenticates itself: `baseline_sha256` is the sha256
  of its canonical JSON with that field removed. The probe verifies it before
  trusting any recorded payload.
- No model weights are loaded (`load_model=False`); only the processor,
  tokenizer, and token identity are resolved. Nothing is written anywhere, and
  the pack-cache root is read for a sha256 inventory only.

The receipt schema carries `commit`, `started_at`, `wall_seconds`, per-split
verdicts, and the cache-root inventory required by the Wave-3 pre-DDP audit
(finding S-2). Strict JSON goes to stdout; progress goes to stderr. Exit code
is nonzero on ANY difference or missing evidence.

Usage:
    conda run -n ms python \
        scripts/probes/coordexp_swift/losses_determinant_equality_probe.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.loader import load_train_config  # noqa: E402
from src.losses.vocab import build_token_vocabulary_groups  # noqa: E402
from src.qwen.loading import load_qwen_components  # noqa: E402
from src.training.cache_workflow import EVAL_SPLIT, TRAIN_SPLIT  # noqa: E402
from src.training.pack_cache import (  # noqa: E402
    PACKING_CACHE_VERSION,
    build_packing_cache_determinants,
)


CHANGE_ID = "standardize-coordexp-swift-supervised-losses"
BASELINE_PATH = Path(
    "openspec/changes/standardize-coordexp-swift-supervised-losses/receipts/"
    "wave-0-determinant-baseline.json"
)
BASELINE_SCHEMA = "coordexp-swift-losses-wave0-determinant-baseline-v1"
# Entry-audit F-10 substitution guard: the ONLY config bound to the two
# published cache targets, and therefore the only config this equality claim
# covers.
PINNED_CONFIG_PATH = (
    "configs/coordexp_swift/smoke/"
    "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000"
    "_accelerate2_ebs2_1step.yaml"
)
CACHE_ROOT = Path(".cache/coordexp_swift/packing")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_root_inventory(root: Path) -> dict[str, Any]:
    """Read-only sha256 inventory of the pack-cache root (audit S-2)."""

    if not root.is_dir():
        return {"root": str(root), "present": False, "files": []}
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "root": str(root),
        "present": True,
        "cache_version": PACKING_CACHE_VERSION,
        "file_count": len(files),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in files),
        "files": files,
    }


def _first_difference(
    recomputed: Any, baseline: Any, path: str = ""
) -> str | None:
    """Report the first differing key-path between two determinant payloads."""

    if isinstance(recomputed, Mapping) and isinstance(baseline, Mapping):
        for key in sorted(set(recomputed) | set(baseline)):
            child = f"{path}.{key}" if path else str(key)
            if key not in recomputed:
                return f"{child}: absent from recomputed payload"
            if key not in baseline:
                return f"{child}: absent from baseline payload"
            found = _first_difference(recomputed[key], baseline[key], child)
            if found is not None:
                return found
        return None
    if isinstance(recomputed, list) and isinstance(baseline, list):
        if len(recomputed) != len(baseline):
            return f"{path}: length {len(recomputed)} != baseline {len(baseline)}"
        for index, (left, right) in enumerate(zip(recomputed, baseline)):
            found = _first_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if recomputed != baseline:
        return f"{path or '<root>'}: {recomputed!r} != baseline {baseline!r}"
    return None


def _load_baseline(findings: list[str]) -> dict[str, Any] | None:
    baseline_path = REPO_ROOT / BASELINE_PATH
    if not baseline_path.is_file():
        findings.append(f"baseline receipt is missing: {BASELINE_PATH}")
        return None
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    schema = baseline.get("schema")
    if schema != BASELINE_SCHEMA:
        findings.append(
            f"baseline schema {schema!r} != expected {BASELINE_SCHEMA!r}"
        )
        return None
    recorded_sha = str(baseline.get("baseline_sha256", ""))
    computed_sha = hashlib.sha256(
        _canonical_json(
            {key: value for key, value in baseline.items() if key != "baseline_sha256"}
        ).encode("utf-8")
    ).hexdigest()
    if recorded_sha != computed_sha:
        findings.append(
            "baseline document failed its own self-authentication: "
            f"baseline_sha256={recorded_sha} but canonical content hashes to "
            f"{computed_sha}"
        )
        return None
    # Entry-audit F-10 substitution guard.
    if baseline.get("config") != PINNED_CONFIG_PATH:
        findings.append(
            "baseline config path does not match the pinned equality scope: "
            f"baseline={baseline.get('config')!r} pinned={PINNED_CONFIG_PATH!r}"
        )
        return None
    missing = [
        split for split in (TRAIN_SPLIT, EVAL_SPLIT) if split not in baseline["splits"]
    ]
    if missing:
        findings.append(f"baseline is missing split payload(s): {missing}")
        return None
    return baseline


def main() -> int:
    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    findings: list[str] = []

    _log(f"[baseline] loading {BASELINE_PATH}")
    baseline = _load_baseline(findings)

    splits: dict[str, Any] = {}
    if baseline is not None:
        config_path = REPO_ROOT / PINNED_CONFIG_PATH
        _log(f"[config] resolving pinned config {PINNED_CONFIG_PATH}")
        resolved = load_train_config(config_path)
        config = resolved.config
        _log("[components] loading processor/tokenizer identity (load_model=False)")
        components = load_qwen_components(config, load_model=False)
        vocab_groups = build_token_vocabulary_groups(
            components.token_identity, tokenizer=components.tokenizer
        )
        split_datasets = {
            TRAIN_SPLIT: config.data.train,
            EVAL_SPLIT: config.data.eval,
        }
        for split, dataset in split_datasets.items():
            if dataset is None:
                findings.append(f"{split}: pinned config declares no dataset split")
                continue
            _log(f"[determinants] recomputing {split}")
            # The config is used EXACTLY as resolved: no model_copy override
            # may enter the determinant payload.
            recomputed = build_packing_cache_determinants(
                config,
                components,
                dataset=dataset,
                split=split,
                vocab_groups=vocab_groups,
            )
            baseline_split = baseline["splits"][split]
            payload_equal = _canonical_json(recomputed) == _canonical_json(
                baseline_split
            )
            recomputed_fingerprint = str(recomputed["aggregate_fingerprint"])
            baseline_fingerprint = str(baseline_split["aggregate_fingerprint"])
            fingerprint_equal = recomputed_fingerprint == baseline_fingerprint
            difference = (
                None if payload_equal else _first_difference(recomputed, baseline_split)
            )
            splits[split] = {
                "verdict": "EQUAL" if payload_equal and fingerprint_equal else "DIFF",
                "payload_equal": payload_equal,
                "fingerprint_equal": fingerprint_equal,
                "recomputed_fingerprint": recomputed_fingerprint,
                "baseline_fingerprint": baseline_fingerprint,
                "first_difference": difference,
            }
            if not payload_equal:
                findings.append(
                    f"{split}: determinant payload differs from the Wave-0 "
                    f"baseline at {difference}"
                )
            if not fingerprint_equal:
                findings.append(
                    f"{split}: aggregate fingerprint {recomputed_fingerprint} != "
                    f"baseline {baseline_fingerprint}"
                )
            _log(f"[determinants] {split}: {splits[split]['verdict']}")

    _log(f"[cache] read-only sha256 inventory of {CACHE_ROOT}")
    inventory = _cache_root_inventory(REPO_ROOT / CACHE_ROOT)
    if not inventory["present"]:
        findings.append(f"pack cache root is absent: {CACHE_ROOT}")

    dirty = _git("status", "--porcelain")
    receipt = {
        "probe": "losses_determinant_equality",
        "task": "5.3",
        "change": CHANGE_ID,
        "commit": _git("rev-parse", "HEAD"),
        "tree_dirty": bool(dirty),
        "baseline_path": str(BASELINE_PATH),
        "baseline_sha256": None if baseline is None else baseline["baseline_sha256"],
        "config": PINNED_CONFIG_PATH,
        "model_loaded": False,
        "cache_materialization_passes": 0,
        "started_at": started_at,
        "wall_seconds": round(time.monotonic() - started_monotonic, 3),
        "splits": splits,
        "cache_root_inventory": inventory,
        "findings": findings,
        "status": "OK" if not findings else "FAILED",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
