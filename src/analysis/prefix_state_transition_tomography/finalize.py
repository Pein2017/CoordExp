from __future__ import annotations

from pathlib import Path
from typing import Any

from .runner import run_from_config
from .status import build_status_report


FINALIZE_STAGES = ("merge", "report", "gallery")


def finalize_if_ready(
    *,
    config_path: Path,
    artifact_root: Path,
    log_root: Path | None = None,
    allow_overwrite: bool = True,
) -> dict[str, Any]:
    status = build_status_report(artifact_root=artifact_root, log_root=log_root)
    if status["stage_status"] == "final_artifacts_present":
        return {
            "action": "already_final",
            "status": status,
            "result": None,
        }
    if status["stage_status"] != "shards_complete_pending_merge":
        return {
            "action": "not_ready",
            "status": status,
            "result": None,
        }
    result = run_from_config(
        Path(config_path),
        stages=FINALIZE_STAGES,
        allow_overwrite=allow_overwrite,
    )
    return {
        "action": "finalized",
        "status": status,
        "result": result,
    }


__all__ = ["FINALIZE_STAGES", "finalize_if_ready"]
