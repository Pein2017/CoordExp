#!/usr/bin/env python3
"""Stamp exact inference provenance sidecars for historical inference runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple, TextIO

_BASE_KEYS = (
    "prompt_policy_fingerprint",
    "decode_policy_fingerprint",
    "model_identity_fingerprint",
)
_RAW_SCORE_KEY = "score_policy"
_RAW_SCORE_NONE = "none"
_SCORE_KEY = "score_policy_fingerprint"
_TRANSITIONAL_PREFIX = "transitional_"
_RAW_ARTIFACT_NAMES = (
    "gt_vs_pred.jsonl",
    "gt_vs_pred_guarded.jsonl",
)
_SCORED_ARTIFACT_NAMES = (
    "gt_vs_pred_scored.jsonl",
    "gt_vs_pred_scored_guarded.jsonl",
)


class _Candidate(NamedTuple):
    payload: dict[str, Any]
    veto_reason: str | None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _is_canonical_fingerprint(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not value.startswith(_TRANSITIONAL_PREFIX)
    )


def _candidate_payloads(*payloads: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    candidates: list[_Candidate] = []
    for payload in payloads:
        carrier_veto = _veto_reason(payload)
        if payload:
            candidates.append(_Candidate(payload=payload, veto_reason=carrier_veto))
        for key in ("inference_provenance", "provenance"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                candidates.append(
                    _Candidate(
                        payload=nested,
                        veto_reason=carrier_veto or _veto_reason(nested),
                    )
                )
    return tuple(candidates)


def _veto_reason(payload: dict[str, Any]) -> str | None:
    if payload.get("comparable") is False:
        return "comparable=false"
    if payload.get("metric_bearing") is False:
        return "metric_bearing=false"
    return None


def _artifact_score_key(path: Path) -> str:
    if path.name in _RAW_ARTIFACT_NAMES:
        return _RAW_SCORE_KEY
    if path.name in _SCORED_ARTIFACT_NAMES:
        return _SCORE_KEY
    raise ValueError(f"unsupported inference artifact name: {path.name}")


def _missing_fields(candidate: dict[str, Any], *, score_key: str) -> list[str]:
    missing = [
        key for key in _BASE_KEYS if not _is_canonical_fingerprint(candidate.get(key))
    ]
    if score_key == _RAW_SCORE_KEY:
        if candidate.get(_RAW_SCORE_KEY) != _RAW_SCORE_NONE:
            missing.append(_RAW_SCORE_KEY)
    elif not _is_canonical_fingerprint(candidate.get(score_key)):
        missing.append(score_key)
    return missing


def _exact_provenance_for_artifact(
    artifact: Path,
    *,
    summary: dict[str, Any],
    resolved_config: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    score_key = _artifact_score_key(artifact)
    best_missing = [*_BASE_KEYS, score_key]
    veto_reason = _veto_reason(summary) or _veto_reason(resolved_config)
    if veto_reason is None:
        for candidate in _candidate_payloads(summary, resolved_config):
            if candidate.veto_reason is not None:
                veto_reason = candidate.veto_reason
                break
    if veto_reason is not None:
        return None, best_missing, veto_reason
    for candidate in _candidate_payloads(summary, resolved_config):
        missing = _missing_fields(candidate.payload, score_key=score_key)
        if missing:
            if len(missing) < len(best_missing):
                best_missing = missing
            continue
        provenance = {key: str(candidate.payload[key]) for key in _BASE_KEYS}
        if score_key == _RAW_SCORE_KEY:
            provenance[_RAW_SCORE_KEY] = _RAW_SCORE_NONE
        else:
            provenance[_SCORE_KEY] = str(candidate.payload[_SCORE_KEY])
        provenance["comparable"] = True
        return provenance, [], None
    return None, best_missing, veto_reason


def _artifact_paths(run_dir: Path) -> tuple[Path, ...]:
    paths = [
        *(run_dir / name for name in _RAW_ARTIFACT_NAMES),
        *(run_dir / name for name in _SCORED_ARTIFACT_NAMES),
    ]
    return tuple(path for path in paths if path.exists())


def _write_sidecar(artifact: Path, payload: dict[str, Any]) -> None:
    sidecar = artifact.with_suffix(artifact.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stamp_run_dir(
    run_dir: Path | str,
    *,
    allow_inspection_stamp: bool = False,
    stdout: TextIO | None = None,
) -> int:
    """Write sidecars only when provenance is exact, or inspection-only by flag."""

    out = stdout if stdout is not None else sys.stdout
    root = Path(run_dir)
    summary = _read_json_object(root / "summary.json")
    resolved_config = _read_json_object(root / "resolved_config.json")
    artifacts = _artifact_paths(root)
    if not artifacts:
        print(f"comparable=false: no supported inference artifacts under {root}", file=out)
        return 1

    planned: list[tuple[Path, dict[str, Any]]] = []
    failures: list[tuple[Path, list[str], str | None]] = []
    for artifact in artifacts:
        provenance, missing, veto_reason = _exact_provenance_for_artifact(
            artifact,
            summary=summary,
            resolved_config=resolved_config,
        )
        if provenance is None:
            failures.append((artifact, missing, veto_reason))
            if allow_inspection_stamp:
                planned.append(
                    (
                        artifact,
                        {
                            "comparable": False,
                            "metric_bearing": False,
                            "missing_provenance": missing,
                            "inspection_reason": (
                                veto_reason
                                or "missing exact canonical inference provenance"
                            ),
                        },
                    )
                )
            continue
        planned.append((artifact, provenance))

    if failures and not allow_inspection_stamp:
        for artifact, missing, veto_reason in failures:
            reason = f"; {veto_reason}" if veto_reason else ""
            print(
                "comparable=false: "
                f"{artifact.name} missing exact provenance fields: {', '.join(missing)}"
                f"{reason}",
                file=out,
            )
        return 1

    for artifact, payload in planned:
        _write_sidecar(artifact, payload)
        print(f"wrote {artifact.name}.provenance.json", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-inspection-stamp",
        action="store_true",
        help="Write comparable=false sidecars when exact provenance is unavailable.",
    )
    args = parser.parse_args(argv)
    return stamp_run_dir(
        args.run_dir,
        allow_inspection_stamp=bool(args.allow_inspection_stamp),
    )


if __name__ == "__main__":
    raise SystemExit(main())
